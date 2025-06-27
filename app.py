import streamlit as st
from PIL import Image
from inference_utils import classify_image

st.set_page_config(page_title="CLIPSat-Foundation-Powered-Satellite-Image-Classifier", layout="centered")
st.title("🛰️ CLIPSat-Foundation-Powered-Satellite-Image-Classifier")

st.write("Upload a satellite image below. The model will predict its category using CLIP ViT-B/32.")

uploaded_image = st.file_uploader("Drag and drop a satellite image here (jpg, png)", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying with CLIP ViT-B/32..."):
        label, confidence, _, _ = classify_image(image)

    st.success(f"Prediction: A satellite image of **{label}** with confidence **{confidence*100:.2f}**%")
else:
    st.info("📤 Please upload an image to start classification.")
