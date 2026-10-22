from io import BytesIO

from PIL import Image, UnidentifiedImageError
from transformers import pipeline

MODEL_NAME = "aaronkantrowitz/ai-human-generated-image-detection"


class AIImageDetector:
    def __init__(self):
        self.classifier = pipeline(
            "image-classification",
            model=MODEL_NAME,
            device=-1,
        )

    def predict(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(BytesIO(image_bytes))
            image.load()
            original_format = image.format or "unknown"
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file is not a valid image.") from exc

        predictions = self.classifier(image, top_k=None)

        scores = {
            item["label"].strip().lower(): float(item["score"])
            for item in predictions
        }

        ai_probability = next(
            (
                score
                for label, score in scores.items()
                if "artificial" in label
                or "ai-generated" in label
                or label == "ai"
                or "fake" in label
            ),
            None,
        )

        if ai_probability is None:
            raise RuntimeError(
                f"Could not identify the AI label in model output: {scores}"
            )

        if ai_probability >= 0.80:
            classification = "likely_ai_generated"
        elif ai_probability >= 0.50:
            classification = "uncertain"
        else:
            classification = "likely_real"

        return {
            "ai_probability": round(ai_probability, 4),
            "classification": classification,
            "image_width": image.width,
            "image_height": image.height,
            "image_format": original_format,
            "disclaimer": (
                "This is a model estimate, not proof of image origin."
            ),
        }
