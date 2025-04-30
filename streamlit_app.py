# import streamlit as st
import joblib
import os
from ocr_utils import extract_text, clean_text

# Load model
model = joblib.load("model/sentiment_model.pkl")

st.title("🧠 Social Media Image Sentiment Classifier")
st.write("Upload an image with text, and we'll detect its sentiment!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image
    img_path = f"temp_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Extract and classify
    text = extract_text(img_path)
    st.subheader("📜 Extracted Text")
    st.write(text)

    if text:
        clean = clean_text(text)
        prediction = model.predict([clean])[0]
        st.subheader("🔍 Sentiment")
        st.success(f"**{prediction}**")

    # Cleanup
    os.remove(img_path)
