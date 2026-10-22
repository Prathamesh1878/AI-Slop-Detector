from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi import File, UploadFile, HTTPException
from src.image_detector import AIImageDetector

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
image_detector = None


from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)

@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.post("/predict")
def predict(request: PredictionRequest):
    return detector.predict(request.text)



@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    global image_detector

    if image_detector is None:
        image_detector = AIImageDetector()

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB.",
        )

    try:
        return image_detector.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Image prediction failed. Check the model output.",
        ) from exc