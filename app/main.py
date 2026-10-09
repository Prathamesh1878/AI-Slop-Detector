from fastapi import FastAPI
from pydantic import BaseModel

from src.detector import AIDetector


app = FastAPI(
    title="AI Slop & AI Content Detector",
    version="1.0.0",
)

detector = AIDetector()


from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)

@app.get("/")
def root():
    return {
        "message": "AI Slop & AI Content Detector API"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    return detector.predict(request.text)
