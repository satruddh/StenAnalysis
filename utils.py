

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as TF

import pydicom
import numpy as np
import json
import os

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
        
        output1 = model1(image_tensor)
        output1 = torch.sigmoid(output1)  

        
        output1_img = TF.to_pil_image(output1.squeeze())  
        
        
        output1_img_rgb = output1_img.convert("RGB")
        output1_tensor = transforms.ToTensor()(output1_img_rgb).unsqueeze(0)

        
        output2 = model2(output1_tensor)
        output2 = torch.sigmoid(output2)

    return output1, output2


def convert_dicom_to_png(dicom_file_path, output_png_path, metadata_output_path=None):
    ds = pydicom.dcmread(dicom_file_path)

    
    metadata = {}
    for key in ["PatientID", "Modality", "StudyDate", "Rows", "Columns", "BitsStored",
                "PhotometricInterpretation", "SamplesPerPixel", "PixelRepresentation"]:
        metadata[key] = str(ds.get(key, "Not Available"))

    if metadata_output_path:
        with open(metadata_output_path, "w") as f:
            json.dump(metadata, f, indent=2)

    
    pixel_array = ds.pixel_array

    
    if len(pixel_array.shape) == 3:
        if pixel_array.shape[0] == 1:
            pixel_array = pixel_array[0]
        elif pixel_array.shape[2] == 3:
            pixel_array = pixel_array[:, :, :3]
        else:
            pixel_array = pixel_array[0]

    
    if pixel_array.dtype == np.uint8:
        pass
    elif pixel_array.dtype == np.uint16:
        pixel_array = (pixel_array / 256).astype(np.uint8)
    else:
        pixel_array = ((pixel_array - pixel_array.min()) /
                      (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

    
    img = Image.fromarray(pixel_array)
    img.save(output_png_path)

    return metadata
