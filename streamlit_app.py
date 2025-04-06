import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
import joblib
import tempfile
from PIL import Image

# Load saved sentiment model
model = joblib.load("sentiment_model.pkl")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Extract text from uploaded image
def extract_text_from_image(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# App UI
st.title("🧠 Social Media Sentiment Classifier (Text from Image)")
st.write("Upload an image with social media text, and get the sentiment (Positive / Negative).")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Text extraction
    with st.spinner("Extracting text..."):
        text = extract_text_from_image(image)

    st.subheader("📝 Extracted Text")
    st.write(text if text else "No text found.")

    # Sentiment classification
    if text:
        preprocessed = preprocess_text(text)
        sentiment = model.predict([preprocessed])[0]
        st.subheader("🔍 Predicted Sentiment")
        st.success(sentiment)
