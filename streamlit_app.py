import streamlit as st
from PIL import Image
import joblib
import easyocr
import numpy as np

# Load trained model
model = joblib.load("sentiment_model.pkl")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# App title
st.title("🧠 Sentiment Analysis from Image")

# Upload an image
uploaded_file = st.file_uploader("Upload an image containing text", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to extract and analyze text
    if st.button("Analyze Sentiment"):
        with st.spinner("Extracting text..."):
            # Convert image to numpy array and extract text
            result = reader.readtext(np.array(image), detail=0)
            extracted_text = " ".join(result)

        st.markdown("**Extracted Text:**")
        st.write(extracted_text)

        if extracted_text.strip() == "":
            st.warning("No text found in the image.")
        else:
            # Preprocess and predict
            cleaned_text = extracted_text.lower()
            prediction = model.predict([cleaned_text])[0]

            # Show result
            if prediction == 0:
                st.success("✅ The comment is **Non-Hateful**")
            else:
                st.error("🚨 The comment is **Hateful**")
