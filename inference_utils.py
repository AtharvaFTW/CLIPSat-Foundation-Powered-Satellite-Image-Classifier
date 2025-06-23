import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel

device="cuda" if torch.cuda.is_available() else "cpu"
model_id="openai/clip-vit-base-patch32"
processor=CLIPProcessor.from_pretrained(model_id)
model=CLIPModel.from_pretrained(model_id).to(device)

CLASS_LABELS=[
    "Annual Crop","Forest","Herbaceous Vegetation","Highway","Industrial","Pasture",
    "Permanent Crop","Residential Area","River","Sea Lake"
]

PROMPT=[f"A satellite image of {label}" for label in CLASS_LABELS]

def classify_image(image):
    image=image.convert("RGB")
    inputs=processor(images=image, text=PROMPT,return_tensors="pt",padding=True).to(device)
    with torch.no_grad():
        outputs=model(**inputs)
        logits_per_image=outputs.logits_per_image
        probs=logits_per_image.softmax(dim=1).squeeze()
    
    predicted_idx=probs.argmax().item()
    return CLASS_LABELS[predicted_idx], probs[predicted_idx].item(),probs.cpu().numpy()