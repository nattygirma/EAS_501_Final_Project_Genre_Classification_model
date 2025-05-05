from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Tuple
import os
import logging
import pickle
import json
import numpy as np
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Movie Genre Classification API")

# Model and tokenizer will be loaded when the application starts
model = None
tokenizer = None
genre_labels = None
mlb = None
metadata = None

class BertGenreClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_output)
        return torch.sigmoid(self.classifier(x))

class MovieDescription(BaseModel):
    text: str

class GenrePrediction(BaseModel):
    genres: Dict[str, float]

class Top3GenrePrediction(BaseModel):
    top_genres: List[Tuple[str, float]]

def predict_top3_genres(text: str) -> List[Tuple[str, float]]:
    """Predict top 3 genres for a given text"""
    if len(text.split()) < 10:
        raise HTTPException(status_code=400, detail="Please enter at least 10 words")
    
    # Tokenize the input text
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=metadata['max_length'],
        return_tensors="pt"
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probs = outputs.cpu().numpy()[0]
    
    # Get top 3 genres
    top3_indices = np.argsort(probs)[-3:][::-1]
    return [(genre_labels[idx], float(probs[idx])) for idx in top3_indices]

@app.on_event("startup")
async def load_model():
    global model, tokenizer, genre_labels, mlb, metadata
    
    try:
        # Load your trained model and tokenizer
        # model_path = os.path.join("app", "model")
        model_path = "app/model"
        logger.info(f"Loading model from: {model_path}")
        
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"Model directory not found at: {model_path}")
        
        # Load metadata
        metadata_path = "app/model/metadata.json"
        # metadata_path = os.path.join(model_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata: {metadata}")
        
        # Load the MultiLabelBinarizer to get genre labels
        mlb_path = "app/model/mlb.pkl"
        # mlb_path = os.path.join(model_path, "mlb.pkl")
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
        genre_labels = mlb.classes_.tolist()
        logger.info(f"Loaded genre labels: {genre_labels}")
        
        # Load the model
        model_file = "app/model/genre_classifier.pt"
        # model_file = os.path.join(model_path, "genre_classifier.pt")
        model_state = torch.load(model_file, map_location=torch.device('cpu'))
        
        # Initialize the model with the correct architecture
        model = BertGenreClassifier(num_labels=len(genre_labels))
        
        # Load the state dict
        model.load_state_dict(model_state)
        model.eval()
        
        # Load the tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        logger.info("Model and tokenizer loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Movie Genre Classification API"}

@app.post("/predict", response_model=GenrePrediction)
async def predict_genres(movie: MovieDescription):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Tokenize the input text
        inputs = tokenizer(
            movie.text,
            padding=True,
            truncation=True,
            max_length=metadata['max_length'],
            return_tensors="pt"
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = outputs.cpu().numpy()[0]
        
        # Convert predictions to dictionary
        genre_scores = {label: float(score) for label, score in sorted(zip(genre_labels, probs), key=lambda x: x[1], reverse=True)}
        
        return GenrePrediction(genres=genre_scores)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/top3", response_model=Top3GenrePrediction)
async def predict_top3(movie: MovieDescription):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        top_genres = predict_top3_genres(movie.text)
        return Top3GenrePrediction(top_genres=top_genres)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 