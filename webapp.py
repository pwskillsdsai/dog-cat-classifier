from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
from typing import Optional
import uvicorn

# Pydantic models for API responses
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    raw_score: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# FastAPI app instance
app = FastAPI(
    title="Dog/Cat Image Classifier",
    description="API for classifying images as containing dogs or cats",
    version="1.0.0"
)

# Global variable to store the loaded model
model = None
MODEL_PATH = "dog_cat_final_model.keras"  # Default model path

@app.on_event("startup")
async def load_ml_model():
    """Load the model on application startup"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model file '{MODEL_PATH}' not found. Please place your model file in the same directory.")
            return
        
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Preprocess the image for prediction
    
    Args:
        img: PIL Image object
        
    Returns:
        Preprocessed image array
    """
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to expected input size
    img = img.resize((128, 128))
    
    # Convert to array and normalize
    img_array = image.img_to_array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Dog/Cat Image Classifier API",
        "endpoints": {
            "/predict": "POST - Upload an image for classification",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict if an uploaded image contains a dog or cat
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction result with confidence score
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs and ensure model file exists."
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read image file
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        img_array = preprocess_image(img)
        
        # Make prediction
        raw_prediction = model.predict(img_array)[0][0]
        
        # Interpret results
        if raw_prediction > 0.5:
            prediction = "DOG"
            confidence = float(raw_prediction)
        else:
            prediction = "CAT"
            confidence = float(1 - raw_prediction)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            raw_score=round(float(raw_prediction), 4)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple images at once
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs and ensure model file exists."
        )
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image"
                })
                continue
            
            # Read and process image
            image_data = await file.read()
            img = Image.open(io.BytesIO(image_data))
            img_array = preprocess_image(img)
            
            # Make prediction
            raw_prediction = model.predict(img_array)[0][0]
            
            # Interpret results
            if raw_prediction > 0.5:
                prediction = "DOG"
                confidence = float(raw_prediction)
            else:
                prediction = "CAT"
                confidence = float(1 - raw_prediction)
            
            results.append({
                "filename": file.filename,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "raw_score": round(float(raw_prediction), 4)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Error processing image: {str(e)}"
            })
    
    return {"results": results}

# Configuration for running the server
if __name__ == "__main__":
    # You can customize these settings
    uvicorn.run(
        "main:app",  # Change "main" to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )