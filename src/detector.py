import joblib
import pandas as pd

from src.features.slop_features import calculate_slop_features
from src.features.slop_score import calculate_slop_score
from src.explanations import generate_explanation


class AIDetector:
    def __init__(self):
        self.ai_model = joblib.load(
            "models/ai_detector.pkl"
        )

        self.vectorizer = joblib.load(
            "models/tfidf_vectorizer.pkl"
        )

        slop_saved = joblib.load(
            "models/slop_detector.pkl"
        )

        self.slop_model = slop_saved["model"]
        self.slop_features = slop_saved["features"]

    def predict(self, text):
        # AI detection
        text_vector = self.vectorizer.transform([text])

        ai_probability = self.ai_model.predict_proba(
            text_vector
        )[0][1]

        # Slop detection
        features = calculate_slop_features(text)

        feature_df = pd.DataFrame([features])
        feature_df = feature_df[self.slop_features]

        slop_probability = self.slop_model.predict_proba(
            feature_df
        )[0][1]

        # Rule-based score
        slop_score = calculate_slop_score(text)

        # Explanation
        explanations = generate_explanation(
            ai_probability,
            slop_probability,
            slop_score,
            features,
        )
        if ai_probability >= 0.80:
            ai_classification = "likely_ai"
        elif ai_probability >= 0.50:
            ai_classification = "uncertain"
        else:
            ai_classification = "likely_human"

        if slop_probability >= 0.80:
            slop_classification = "high"
        elif slop_probability >= 0.50:
            slop_classification = "moderate"
        else:
            slop_classification = "low"

        return {
            "ai_probability": round(
                float(ai_probability), 4
            ),
            "slop_probability": round(
                float(slop_probability), 4
            ),
            "slop_score": slop_score,
            "explanations": explanations,
            "ai_classification": ai_classification,
            "slop_classification": slop_classification,
        }