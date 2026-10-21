const analyzeButton = document.getElementById("analyzeButton");
const textInput = document.getElementById("textInput");
const results = document.getElementById("results");

analyzeButton.addEventListener("click", async () => {
    const text = textInput.value.trim();

    if (!text) {
        results.innerHTML = "<p>Please enter some text.</p>";
        return;
    }

    results.innerHTML = "<p>Analyzing...</p>";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (!response.ok) {
            results.innerHTML = `<p>Request failed. Please check your input.</p>`;
            return;
        }

        results.innerHTML = `
            <h2>Results</h2>

            <div class="result-card">
                <h3>AI Content</h3>
                <p class="result-value">
                    ${(data.ai_probability * 100).toFixed(2)}%
                </p>
                <p>Classification: <strong>${data.ai_classification}</strong></p>
            </div>

            <div class="result-card">
                <h3>AI Slop</h3>
                <p class="result-value">
                    ${(data.slop_probability * 100).toFixed(2)}%
                </p>
                <p>Classification: <strong>${data.slop_classification}</strong></p>
                <p>Slop Score: <strong>${data.slop_score}/100</strong></p>
            </div>

            <div class="result-card">
                <h3>Explanation</h3>
                <ul>
                    ${data.explanations
                        .map(reason => `<li>${reason}</li>`)
                        .join("")}
                </ul>
            </div>
        `;
    } catch (error) {
        results.innerHTML = `
            <p>Could not connect to the detector API.</p>
        `;
    }
});