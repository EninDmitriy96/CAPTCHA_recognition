from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import load_learner, PILImage
import torch
from torchvision import transforms
from pydantic import BaseModel
from typing import List
import pathlib
from fastapi.middleware.cors import CORSMiddleware
import torch.nn as nn
from torchvision.models import resnet50

class CaptchaSolver(nn.Module):
    def __init__(self):
        super(CaptchaSolver, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(2048, 5 * 62)
        
    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out.view(-1, 5, 62)

captcha_model = CaptchaSolver()

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load models
seq_models = [load_learner(f"../../../models/seq_captcha_models/captcha_model_char_{i + 1}.pkl") for i in range(5)]
state_dict = torch.load("../../../models/captcha_models/captcha_solver.pth", map_location=torch.device('cpu'))
captcha_model.load_state_dict(state_dict)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
SEQ_LEN = 5  # Length of captcha

# Helper for sequential prediction
def sequential_predict(learn_list, image):
    predicted_labels = []
    for learn in learn_list:
        pred, _, _ = learn.predict(image)
        predicted_labels.append(pred)
    return "".join(predicted_labels)

# Helper for whole captcha prediction
def whole_captcha_predict(model, image):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0)
        output = model(image)
        predicted_text = ""
        for i in range(SEQ_LEN):
            char_idx = output[0][i].argmax().item()
            predicted_text += CHARACTERS[char_idx]
        return predicted_text

@app.post("/predict/")
async def predict_captcha(image: UploadFile = File(...)):
    print("Endpoint hit")
    img = PILImage.create(await image.read())
    # Sequential prediction
    seq_prediction = sequential_predict(seq_models, img)
    
    # Whole captcha prediction
    whole_prediction = whole_captcha_predict(captcha_model, img)
    return {"sequential_prediction": seq_prediction, "whole_prediction": whole_prediction}
