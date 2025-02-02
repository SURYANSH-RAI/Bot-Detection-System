from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from bot_detection_analysis import BotDetectionSystem  # Your rule-based analysis file
import pandas as pd
import torch
from bot_detection_ml import BotDetectionTrainer  # Your ML model file
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:8000",
    "http://localhost:5173",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rule-based system
detector = BotDetectionSystem()

# Initialize ML model
model = BotDetectionTrainer()
model.load_model('bot_detection_model.pt')

class AnalysisRequest(BaseModel):
    csvData: List[Dict]
    socialLinks: List[str]
    profileLinks: List[str]

@app.post("/api/rule-based-analysis")
async def analyze_with_rules(request: AnalysisRequest):
    try:
        # Process the CSV data
        data = pd.DataFrame(request.csvData)
        
        # Analyze using rule-based system
        results = detector.analyze_profile(data)
        
        # Format response
        return {
            "botProbability": results['bot_probability'],
            "contentPatterns": {
                "spamScore": results['content_patterns']['spam_score'],
                "sentimentScore": results['content_patterns']['sentiment_score']
            },
            "behavioralMetrics": {
                "postingFrequency": results['behavioral_metrics']['posting_frequency'],
                "engagementRate": results['behavioral_metrics']['engagement_rate']
            },
            "ruleMatches": results['matched_rules_count']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/ml-analysis")
async def analyze_with_ml(request: AnalysisRequest):
    print("helllo")
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
    
