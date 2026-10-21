from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from src.detector import AIDetector


app = FastAPI(
    title="AI Slop & AI Content Detector",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

detector = AIDetector()


from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)

@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.post("/predict")
def predict(request: PredictionRequest):
    return detector.predict(request.text)
