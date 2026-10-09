def generate_explanation(
    ai_probability,
    slop_probability,
    slop_score,
    features,
):
    explanations = []

    # AI signal
    if ai_probability >= 0.80:
        explanations.append(
            "AI probability is high."
        )
    elif ai_probability >= 0.50:
        explanations.append(
            "AI probability is moderate."
        )
    else:
        explanations.append(
            "AI probability is low."
        )

    # Slop probability
    if slop_probability >= 0.80:
        explanations.append(
            "Slop probability is high."
        )
    elif slop_probability >= 0.50:
        explanations.append(
            "Slop probability is moderate."
        )
    else:
        explanations.append(
            "Slop probability is low."
        )

    # Feature-based reasons
    if features["generic_sentence_ratio"] >= 0.30:
        explanations.append(
            "Generic sentence patterns were detected."
        )

    if features["vague_phrase_count"] >= 2:
        explanations.append(
            "Vague or broad phrasing was detected."
        )

    if features["generic_recommendation_count"] >= 2:
        explanations.append(
            "Generic recommendation language was detected."
        )

    if features["repeated_ngram_ratio"] >= 0.05:
        explanations.append(
            "Repeated phrases or word patterns were detected."
    )

    if features["sentence_redundancy"] >= 0.18:
        explanations.append(
            "Some sentences appear semantically repetitive."
        )

    if features["buzzword_count"] >= 3:
        explanations.append(
            "Multiple broad or buzzword-heavy terms were detected."
        )

    

    # Fallback
    if len(explanations) == 2:
        explanations.append(
            "No strong slop indicators were detected."
        )

    return explanations