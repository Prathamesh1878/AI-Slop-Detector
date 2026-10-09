from slop_features import calculate_slop_features
from slop_score import calculate_slop_score


def explain_slop(text):
    features = calculate_slop_features(text)
    score = calculate_slop_score(text)

    reasons = []

    if features["filler_phrase_count"] > 0:
        reasons.append(
            f"Contains {features['filler_phrase_count']} "
            "generic/filler phrase(s)."
        )

    if features["buzzword_density"] >= 0.05:
        reasons.append(
            "Uses a relatively high density of buzzwords."
        )

    if features["repetition_ratio"] >= 0.20:
        reasons.append(
            "Contains a relatively high amount of repeated vocabulary."
        )

    if features["avg_sentence_length"] > 25:
        reasons.append(
            "Uses unusually long sentences."
        )

    if (
        features["question_ratio"] == 0
        and features["word_count"] > 100
    ):
        reasons.append(
            "Uses a formulaic statement-heavy structure."
        )

    if score < 20:
        level = "Low"
    elif score < 50:
        level = "Moderate"
    else:
        level = "High"

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
    }


if __name__ == "__main__":
    example = """
    In today's rapidly changing world, it is important to leverage
    innovative solutions to unlock unprecedented opportunities and
    drive meaningful transformation.
    """

    result = explain_slop(example)

    print("=" * 60)
    print("SLOP ANALYSIS")
    print("=" * 60)

    print()
    print(f"Score: {result['score']}/100")
    print(f"Level: {result['level']}")

    print()
    print("Reasons:")

    for reason in result["reasons"]:
        print(f"- {reason}")