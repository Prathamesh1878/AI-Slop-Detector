from .slop_features import calculate_slop_features


def calculate_slop_score(text):
    """
    Calculate a slop score from 0 to 100.

    Higher scores indicate stronger signals of generic,
    repetitive, vague, boilerplate, or low-information writing.
    """

    features = calculate_slop_features(text)

    score = 0.0

    # 1. Boilerplate language
    score += min(
        features["boilerplate_phrase_count"] * 8,
        20,
    )

    # 2. AI-style disclaimer language
    score += min(
        features["ai_disclaimer_count"] * 10,
        20,
    )

    # 3. Generic recommendation language
    score += min(
        features["generic_recommendation_count"] * 5,
        15,
    )

    # 4. Repeated phrases / n-grams
    score += min(
        features["repeated_ngram_ratio"] * 100,
        20,
    )

    # 5. Vague language
    score += min(
        features["vague_phrase_density"] * 1.2,
        15,
    )

    # 6. Generic filler phrases
    score += min(
        features["filler_phrase_count"] * 4,
        10,
    )

    # 7. Buzzword density
    score += min(
        features["buzzword_density"] * 50,
        10,
    )

    # 8. Sentence redundancy
    if features["sentence_redundancy"] >= 0.35:
        score += 15
    elif features["sentence_redundancy"] >= 0.25:
        score += 10
    elif features["sentence_redundancy"] >= 0.18:
        score += 5
    elif features["sentence_redundancy"] >= 0.12:
        score += 3

    if features["generic_sentence_ratio"] >= 0.60:
        score += 6
    elif features["generic_sentence_ratio"] >= 0.45:
        score += 4
    elif features["generic_sentence_ratio"] >= 0.30:
        score += 2
        
    # 9. Combined boilerplate signal
    combined_generic_signal = (
        features["ai_disclaimer_count"]
        + features["generic_recommendation_count"]
        + features["boilerplate_phrase_count"]
    )

    if combined_generic_signal >= 4:
        score += 8

    elif combined_generic_signal >= 3:
        score += 5

    # 10. Low-information interaction
    if features["word_count"] >= 120:

        low_concrete_signal = (
            features["number_density"] < 0.005
            and features["named_entity_signal"] < 0.02
        )

        vague_language = (
            features["vague_phrase_density"] >= 1.5
        )

        generic_language = (
            features["generic_recommendation_density"] >= 1.0
            or features["ai_disclaimer_density"] >= 1.0
            or features["boilerplate_density"] >= 1.0
        )

        if (
            low_concrete_signal
            and vague_language
            and generic_language
        ):
            score += 5

    return round(min(score, 100), 2)


if __name__ == "__main__":

    examples = [
        """
        This innovative and transformative solution
        provides a seamless experience.

        It is important to note that this solution
        can provide many meaningful benefits.

        Depending on the situation, it may help
        organizations unlock exciting opportunities.

        I recommend speaking with a professional
        if you have specific questions.
        """,

        """
        Python is a programming language commonly used
        for data analysis, machine learning, and web
        development.
        """,
    ]

    for index, text in enumerate(examples, start=1):

        score = calculate_slop_score(text)

        print(f"Example {index}")
        print(f"Slop score: {score}")
        print()