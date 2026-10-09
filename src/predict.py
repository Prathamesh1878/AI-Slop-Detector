from detector import AIDetector


detector = AIDetector()


if __name__ == "__main__":
    text = input("Enter text to analyze:\n\n")

    result = detector.predict(text)

    print()
    print("=" * 50)
    print("AI CONTENT & SLOP DETECTOR")
    print("=" * 50)
    print()

    print(
        f"AI probability:   "
        f"{result['ai_probability'] * 100:.2f}%"
    )

    print(
        f"Slop probability: "
        f"{result['slop_probability'] * 100:.2f}%"
    )

    print(
        f"Rule-based score: "
        f"{result['slop_score']:.2f}/100"
    )

    print()
    print("Explanation:")

    for explanation in result["explanations"]:
        print(f"• {explanation}")