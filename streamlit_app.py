import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Set page title
st.title("Dog vs Cat Classifier")

# Load the model
@st.cache_resource
def load_classification_model():
    return load_model('dog_cat_final_model.keras')

try:
    model = load_classification_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)
    
    # Preprocess the image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(img_array)[0][0]
        
        # Display result
        if prediction > 0.5:
            st.success(f"This is a Dog! (Confidence: {prediction:.2f})")
        else:
            st.success(f"This is a Cat! (Confidence: {1-prediction:.2f})")
        
        # Display probability bar
        st.progress(float(prediction) if prediction > 0.5 else float(1-prediction))
        
        # Display probability distribution
        st.bar_chart({
            "Cat": 1-prediction,
            "Dog": prediction
        })

# Instructions
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Upload an image of a dog or cat\n"
    "2. Click the 'Predict' button\n"
    "3. View the prediction result"
)

# About
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a CNN model trained on a dogs and cats dataset. "
    "The model was trained using TensorFlow/Keras."
)