import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict/"

st.title("Captcha Recognition System")
st.write("Upload a captcha image to see predictions from two approaches:")

uploaded_file = st.file_uploader("Upload Captcha", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image with specified width
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Captcha", width=200)  # Adjust width as needed
    
    # Reset file pointer
    uploaded_file.seek(0)

    # Call the FastAPI backend
    with st.spinner("Making predictions..."):
        files = {"image": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
        response = requests.post(API_URL, files=files)
    if response.status_code == 200:
        predictions = response.json()
        st.subheader("Predictions:")
        st.write(f"**Sequential Prediction:** {predictions['sequential_prediction']}")
        st.write(f"**Whole Captcha Prediction:** {predictions['whole_prediction']}")
    else:
        st.error("Error in prediction. Please try again.")
