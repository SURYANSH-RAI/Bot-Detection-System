from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
import pandas as pd
from bot_detection_ml import BotDetectionTrainer  # Your ML model file

app = FastAPI()

# Initialize ML model
model = BotDetectionTrainer()
model.load_model('bot_detection_model.pt')

class AnalysisRequest(BaseModel):
    csvData: List[Dict]
    socialLinks: List[str]
    profileLinks: List[str]

@app.post("/api/ml-analysis")
async def analyze_with_ml(request: AnalysisRequest):
    try:
        # Process the CSV data
        data = pd.DataFrame(request.csvData)
        
        # Get predictions from ML model
        predictions = model.predict(data)
        
        # Format response
        return {
            "botProbability": float(predictions[0]),
            "contentPatterns": {
                "spamScore": float(predictions[1]),
                "sentimentScore": float(predictions[2])
            },
            "behavioralMetrics": {
                "postingFrequency": int(predictions[3]),
                "engagementRate": float(predictions[4])
            },
            "mlConfidence": float(predictions[5])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))