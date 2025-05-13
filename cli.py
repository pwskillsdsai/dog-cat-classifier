import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

def predict_image(img_path, model_path):
    """
    Predict if an image contains a dog or cat
    
    Args:
        img_path: Path to the image file
        model_path: Path to the trained model (.h5 file)
    """
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return
    
    try:
        # Load model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        
        # Load and preprocess image
        print(f"Processing image {img_path}...")
        img = Image.open(img_path)
        img = img.resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Display result
        if prediction > 0.5:
            print(f"Prediction: DOG (Confidence: {prediction:.2f})")
        else:
            print(f"Prediction: CAT (Confidence: {1-prediction:.2f})")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict if an image contains a dog or cat')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='dog_cat_final_model.keras', 
                        help='Path to the model file (default: dog_cat_cnn_model.h5)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run prediction
    predict_image(args.image_path, args.model)

    # python cli_inference.py path/to/your/image.jpg
    # python cli_inference.py path/to/your/image.jpg --model path/to/your/model.keras