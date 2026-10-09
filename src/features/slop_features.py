import re
from collections import Counter

# ---------------------------------------------------------
# V3 phrase dictionaries
# ---------------------------------------------------------

BUZZWORDS = {
    "innovative",
    "transformative",
    "revolutionary",
    "cutting-edge",
    "seamless",
    "robust",
    "scalable",
    "leverage",
    "empower",
    "unlock",
    "unprecedented",
    "meaningful",
    "holistic",
    "synergy",
    "paradigm",
    "game-changing",
    "next-generation",
}

FILLER_PHRASES = [
    "in today's world",
    "in today's rapidly changing world",
    "it is important to note",
    "it is worth noting",
    "in conclusion",
    "at the end of the day",
    "when it comes to",
    "plays a crucial role",
    "plays an important role",
    "a wide range of",
    "not only",
    "but also",
    "moving forward",
]

BOILERPLATE_PHRASES = [
    "it is important to consult",
    "consult a doctor",
    "consult your doctor",
    "speak with a doctor",
    "speak with a healthcare provider",
    "seek medical attention",
    "seek professional advice",
    "talk to a financial advisor",
    "speak with a financial advisor",
    "do your own research",
    "it is always a good idea",
    "if you have any questions",
    "if you have specific questions",
    "for more information",
    "depending on your situation",
    "depending on the circumstances",
]

VAGUE_PHRASES = [
    "various factors",
    "many factors",
    "many different",
    "a number of",
    "a variety of",
    "in many cases",
    "in some cases",
    "can help",
    "may help",
    "can be useful",
    "may be useful",
    "plays a role",
    "plays an important role",
    "offers several benefits",
    "many benefits",
    "exciting opportunities",
    "meaningful benefits",
    "across a wide range",
    "depending on the situation",
]

AI_DISCLAIMER_PHRASES = [
    "i can't provide",
    "i cannot provide",
    "i don't have access",
    "i do not have access",
    "i'm unable to",
    "i am unable to",
    "i cannot assist",
    "i can't assist",
    "i cannot comment",
    "i can't comment",
    "without knowing more",
    "without more information",
    "without knowing your",
    "i recommend speaking with",
    "i recommend consulting",
    "i recommend contacting",
    "you should consult",
    "you should speak with",
    "you may want to consult",
    "you may want to speak with",
]

GENERIC_RECOMMENDATION_PHRASES = [
    "it is important to",
    "it's important to",
    "it is a good idea to",
    "it's a good idea to",
    "it may be helpful to",
    "it can be helpful to",
    "you may want to",
    "you should consider",
    "one option is to",
    "another option is to",
    "for more information",
    "do your own research",
    "seek professional advice",
]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9'-]+\b", text.lower())


def calculate_repetition_ratio(words):
    if not words:
        return 0.0

    counts = Counter(words)

    repeated_words = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated_words / len(words)


def calculate_repeated_ngram_ratio(words, n=3):
    if len(words) < n:
        return 0.0

    ngrams = [
        tuple(words[i:i + n])
        for i in range(len(words) - n + 1)
    ]

    counts = Counter(ngrams)

    repeated_count = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated_count / len(ngrams)


def calculate_sentence_redundancy(sentences):
    if len(sentences) < 2:
        return 0.0

    sentence_words = [
        set(tokenize(sentence))
        for sentence in sentences
    ]

    similarities = []

    for i in range(len(sentence_words)):
        for j in range(i + 1, len(sentence_words)):
            a = sentence_words[i]
            b = sentence_words[j]

            if not a or not b:
                continue

            intersection = len(a & b)
            union = len(a | b)

            if union > 0:
                similarities.append(intersection / union)

    if not similarities:
        return 0.0

    return sum(similarities) / len(similarities)

def calculate_generic_sentence_ratio(sentences):
    if not sentences:
        return 0.0

    generic_patterns = [
        r"\bit is\b",
        r"\bit's\b",
        r"\bthis is\b",
        r"\bthis means\b",
        r"\bthis can\b",
        r"\bthis may\b",
        r"\bthis helps\b",
        r"\bthis allows\b",
        r"\bthis is because\b",
        r"\bso that\b",
        r"\bin other words\b",
        r"\bfor example\b",
        r"\bhowever\b",
        r"\bon the other hand\b",
        r"\bin general\b",
    ]

    generic_sentences = 0

    for sentence in sentences:
        sentence_lower = sentence.lower()

        if any(
            re.search(pattern, sentence_lower)
            for pattern in generic_patterns
        ):
            generic_sentences += 1

    return generic_sentences / len(sentences)

def count_phrase_occurrences(text, phrases):
    text = text.lower()

    return sum(
        text.count(phrase)
        for phrase in phrases
    )


# ---------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------

def calculate_slop_features(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    lower_text = text.lower()

    words = tokenize(text)
    word_count = len(words)

    # Sentence splitting
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    sentence_count = len(sentences)

    # -----------------------------------------------------
    # Basic statistics
    # -----------------------------------------------------

    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count
    else:
        avg_sentence_length = 0.0

    # -----------------------------------------------------
    # Buzzwords
    # -----------------------------------------------------

    buzzword_count = sum(
        1
        for word in words
        if word in BUZZWORDS
    )

    if word_count > 0:
        buzzword_density = buzzword_count / word_count
    else:
        buzzword_density = 0.0

    # -----------------------------------------------------
    # Phrase features
    # -----------------------------------------------------

    filler_phrase_count = count_phrase_occurrences(
        lower_text,
        FILLER_PHRASES,
    )

    boilerplate_phrase_count = count_phrase_occurrences(
        lower_text,
        BOILERPLATE_PHRASES,
    )

    ai_disclaimer_count = count_phrase_occurrences(
        lower_text,
        AI_DISCLAIMER_PHRASES,
    )

    generic_recommendation_count = count_phrase_occurrences(
        lower_text,
        GENERIC_RECOMMENDATION_PHRASES,
    )

    vague_phrase_count = count_phrase_occurrences(
        lower_text,
        VAGUE_PHRASES,
    )

    # -----------------------------------------------------
    # Phrase densities
    # -----------------------------------------------------

    if word_count > 0:
        boilerplate_density = (
            boilerplate_phrase_count / word_count * 100
        )

        ai_disclaimer_density = (
            ai_disclaimer_count / word_count * 100
        )

        generic_recommendation_density = (
            generic_recommendation_count / word_count * 100
        )

        vague_phrase_density = (
            vague_phrase_count / word_count * 100
        )
    else:
        boilerplate_density = 0.0
        ai_disclaimer_density = 0.0
        generic_recommendation_density = 0.0
        vague_phrase_density = 0.0

    # -----------------------------------------------------
    # Repetition
    # -----------------------------------------------------

    repetition_ratio = calculate_repetition_ratio(words)

    repeated_ngram_ratio = calculate_repeated_ngram_ratio(
        words,
        n=3,
    )

    sentence_redundancy = calculate_sentence_redundancy(
        sentences
    )

    generic_sentence_ratio = calculate_generic_sentence_ratio(
    sentences
    )

    # -----------------------------------------------------
    # Punctuation / structure
    # -----------------------------------------------------

    if sentence_count > 0:
        question_count = text.count("?")
        exclamation_count = text.count("!")

        question_ratio = question_count / sentence_count
        exclamation_ratio = exclamation_count / sentence_count
    else:
        question_ratio = 0.0
        exclamation_ratio = 0.0

    # -----------------------------------------------------
    # Concrete information signals
    # -----------------------------------------------------

    number_count = len(
        re.findall(r"\b\d+(?:\.\d+)?\b", text)
    )

    if word_count > 0:
        number_density = number_count / word_count
    else:
        number_density = 0.0

    # Simple named-entity proxy:
    # count capitalized words that aren't at sentence starts
    named_entity_count = 0

    for index, word in enumerate(
        re.findall(r"\b[A-Z][a-zA-Z'-]+\b", text)
    ):
        if index > 0:
            named_entity_count += 1

    if word_count > 0:
        named_entity_signal = named_entity_count / word_count
    else:
        named_entity_signal = 0.0

    # -----------------------------------------------------
    # Return all features
    # -----------------------------------------------------

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,

        "buzzword_count": buzzword_count,
        "buzzword_density": buzzword_density,

        "filler_phrase_count": filler_phrase_count,

        "boilerplate_phrase_count": boilerplate_phrase_count,
        "boilerplate_density": boilerplate_density,

        "ai_disclaimer_count": ai_disclaimer_count,
        "ai_disclaimer_density": ai_disclaimer_density,

        "generic_recommendation_count":
            generic_recommendation_count,

        "generic_recommendation_density":
            generic_recommendation_density,

        "vague_phrase_count": vague_phrase_count,
        "vague_phrase_density": vague_phrase_density,

        "repetition_ratio": repetition_ratio,
        "repeated_ngram_ratio": repeated_ngram_ratio,
        "sentence_redundancy": sentence_redundancy,
        "generic_sentence_ratio": generic_sentence_ratio,

        "question_ratio": question_ratio,
        "exclamation_ratio": exclamation_ratio,

        "number_density": number_density,
        "named_entity_signal": named_entity_signal,
    }


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    example = """
    This innovative and transformative solution
    provides a seamless experience.

    It is important to note that this solution
    can provide many meaningful benefits.

    Depending on the situation, it may help
    organizations unlock exciting opportunities.

    I recommend speaking with a professional
    if you have specific questions.
    """

    features = calculate_slop_features(example)

    print("Slop features:")

    for name, value in features.items():
        print(f"{name}: {value}")