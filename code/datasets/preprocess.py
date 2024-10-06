import os
from PIL import Image
from tqdm import tqdm


def remove_corrupted_images(directory):
    all_images = [os.path.join(directory, file) for file in os.listdir(directory) if
                  file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    print(f"Total number of images before cleanup: {len(all_images)}")

    corrupted_images = []

    for image_path in tqdm(all_images, desc="Checking images", unit="image"):
        try:
            with Image.open(image_path) as img:
                img.verify()
        except (IOError, SyntaxError):
            corrupted_images.append(image_path)

    for corrupted_image in corrupted_images:
        os.remove(corrupted_image)

    valid_images = [os.path.join(directory, file) for file in os.listdir(directory) if
                    file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    print(f"Total number of images after cleanup: {len(valid_images)}")
    print(f"Number of corrupted images removed: {len(corrupted_images)}")


directory = '../../data/captchas_dataset_1/Large_Captcha_Dataset'
remove_corrupted_images(directory)
