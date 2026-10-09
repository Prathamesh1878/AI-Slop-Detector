import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_PATH = "models/ai_detector.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def train_baseline():
    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples: {len(test_df)}")

    X_train = train_df["text"]
    y_train = train_df["ai_label"]

    X_validation = validation_df["text"]
    y_validation = validation_df["ai_label"]

    X_test = test_df["text"]
    y_test = test_df["ai_label"]

    print()
    print("Creating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_validation_tfidf = vectorizer.transform(X_validation)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Training feature matrix: {X_train_tfidf.shape}")
    print(f"Validation feature matrix: {X_validation_tfidf.shape}")
    print(f"Test feature matrix: {X_test_tfidf.shape}")

    print()
    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    print("Training complete!")

    print()
    print("Validation evaluation...")

    validation_predictions = model.predict(X_validation_tfidf)

    print(
        classification_report(
            y_validation,
            validation_predictions,
            target_names=["Human", "AI"],
        )
    )

    print("Test evaluation...")

    test_predictions = model.predict(X_test_tfidf)

    test_accuracy = accuracy_score(y_test, test_predictions)

    print()
    print(f"Test accuracy: {test_accuracy:.4f}")

    print()
    print("Test classification report:")

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Human", "AI"],
        )
    )

    print("Test confusion matrix:")

    print(confusion_matrix(y_test, test_predictions))

    print()
    print("Saving model...")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print()
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_baseline()