import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
import json
import pickle

device="cuda" if torch.cuda.is_available() else "cpu"
model_id="openai/clip-vit-base-patch32"
processor=CLIPProcessor.from_pretrained(model_id)
model=CLIPModel.from_pretrained(model_id).to(device)

with open("classifier.pkl","rb") as f:
    classifier=pickle.load(f)

with open("label_to_index.json","r") as f:
    label_to_index=json.load(f)
    index_to_label={v:k for k,v in label_to_index.items()}


def classify_image(image):
    inputs=processor(images=image,return_tensors="pt").to(device)
    with torch.no_grad():
        image_features=model.get_image_features(**inputs)
        embedding=image_features.cpu().numpy().flatten().reshape(1,-1)
    
    pred=classifier.predict(embedding)[0]
    proba=classifier.predict_proba(embedding)[0]
    
    label=index_to_label[pred]
    confidence=proba[pred]
    
    return label,confidence,proba,list(index_to_label.values())
