import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Constants
SEQ_LEN = 5
NUM_CLASSES = 62
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Functions for character encoding
def char_to_onehot(char):
    onehot = np.zeros(NUM_CLASSES, dtype=np.float32)
    onehot[CHARACTERS.index(char)] = 1.0
    return onehot

def text_to_onehot(text):
    matrix = np.zeros((SEQ_LEN, NUM_CLASSES), dtype=np.float32)
    for i, char in enumerate(text):
        matrix[i] = char_to_onehot(char)
    return matrix.flatten()

# Dataset class
class CaptchaDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        
        print("Loading dataset...")
        for fname in tqdm(os.listdir(data_path), desc="Processing files"):
            img_path = os.path.join(data_path, fname)
            label = fname.split('.')[0]
            try:
                with Image.open(img_path) as img:
                    img.verify()
                self.samples.append((img_path, label))
            except (IOError, SyntaxError):
                print(f"Skipping corrupted file: {fname}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_onehot = text_to_onehot(label)
        return image, torch.tensor(label_onehot, dtype=torch.float32)

# Model definition
class CaptchaSolver(nn.Module):
    def __init__(self):
        super(CaptchaSolver, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(2048, SEQ_LEN * NUM_CLASSES)
        
    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out.view(-1, SEQ_LEN, NUM_CLASSES)

# Functions for training and validation
def validate(model, loader, criterion, device, seq_len, characters):
    model.eval()
    total_loss = 0.0
    total_chars = 0
    correct_chars = 0
    total_captchas = 0
    correct_captchas = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, seq_len, len(characters))
            
            outputs = model(images)
            
            loss = sum(criterion(outputs[:, i, :], labels[:, i, :].argmax(dim=1)) for i in range(seq_len))
            total_loss += loss.item()
            
            for i in range(seq_len):
                predictions = outputs[:, i, :].argmax(dim=1)
                targets = labels[:, i, :].argmax(dim=1)
                correct_chars += (predictions == targets).sum().item()
                total_chars += predictions.size(0)
            
            for i in range(outputs.size(0)):
                predicted_text = "".join(
                    characters[outputs[i, j, :].argmax().item()] for j in range(seq_len)
                )
                true_text = "".join(
                    characters[labels[i, j, :].argmax().item()] for j in range(seq_len)
                )
                if predicted_text == true_text:
                    correct_captchas += 1
                total_captchas += 1
    
    char_accuracy = correct_chars / total_chars
    captcha_accuracy = correct_captchas / total_captchas
    return total_loss / len(loader), char_accuracy, captcha_accuracy

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, seq_len, characters, model_path="captcha_solver.pth"):
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, seq_len, len(characters))
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = sum(criterion(outputs[:, i, :], labels[:, i, :].argmax(dim=1)) for i in range(seq_len))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        val_loss, char_acc, captcha_acc = validate(model, val_loader, criterion, device, seq_len, characters)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Char Accuracy: {char_acc:.4f}, Captcha Accuracy: {captcha_acc:.4f}")

# Main block
if __name__ == "__main__":
    DATASET_PATH = '/kaggle/input/large-captcha-dataset/Large_Captcha_Dataset'
    BATCH_SIZE = 32
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CaptchaDataset(DATASET_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = CaptchaSolver().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train(model, train_loader, val_loader, criterion, optimizer, epochs=20, device=device, seq_len=SEQ_LEN, characters=CHARACTERS)
