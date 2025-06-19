

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as TF

MODEL1_PATH = "./model1_scripted.pt"
MODEL2_PATH = "./model2_scripted.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = torch.jit.load(MODEL1_PATH).to(device).eval()
model2 = torch.jit.load(MODEL2_PATH).to(device).eval()


def preprocess_image(image_path):
    """Preprocess the input image for model1."""
    inp = Image.open(image_path).convert("RGB")
    tr = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    inp = tr(inp).unsqueeze(0)
    return inp

def run_inference(image_tensor):
    """Run inference through both models with reprocessing for model2."""
    with torch.no_grad():
        # Stage 1: Model 1 Inference
        output1 = model1(image_tensor)
        output1 = torch.sigmoid(output1)  # Single-channel mask in [0, 1]

        # Convert to PIL image to mimic save-load behavior
        output1_img = TF.to_pil_image(output1.squeeze())  # Converts to single-channel PIL image
        
        # Reprocess for model2: Convert to 3-channel RGB
        output1_img_rgb = output1_img.convert("RGB")
        output1_tensor = transforms.ToTensor()(output1_img_rgb).unsqueeze(0)

        # Stage 2: Model 2 Inference
        output2 = model2(output1_tensor)
        output2 = torch.sigmoid(output2)

    return output1, output2


# image_path = "./26.png"

# img = preprocess_image(image_path)
# # print(img.shape)
# x,y = run_inference(img)
