from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "AI Slop & AI Content Detector API"
    )


def test_predict_valid_text():
    response = client.post(
        "/predict",
        json={
            "text": (
                "Artificial intelligence is transforming "
                "the way we live and work."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "ai_probability" in data
    assert "ai_classification" in data
    assert "slop_probability" in data
    assert "slop_classification" in data
    assert "slop_score" in data
    assert "explanations" in data


def test_predict_text_too_short():
    response = client.post(
        "/predict",
        json={"text": "hi"},
    )

    assert response.status_code == 422


def test_predict_text_too_long():
    response = client.post(
        "/predict",
        json={"text": "a" * 10001},
    )

    assert response.status_code == 422