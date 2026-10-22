"use strict";

// ===============================
// GET HTML ELEMENTS
// ===============================

const analyzeButton = document.getElementById("analyzeButton");
const textInput = document.getElementById("textInput");
const results = document.getElementById("results");

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");
const imageDetails = document.getElementById("imageDetails");
const analyzeImageButton = document.getElementById("analyzeImageButton");
const imageResults = document.getElementById("imageResults");

const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const themeLabel = document.getElementById("themeLabel");
const charCount = document.getElementById("charCount");


// ===============================
// TEXT DETECTOR
// ===============================

analyzeButton.addEventListener("click", async function () {
    const text = textInput.value.trim();

    if (text.length < 10) {
        results.textContent = "Please enter at least 10 characters.";
        return;
    }

    if (text.length > 10000) {
        results.textContent = "Text must not exceed 10,000 characters.";
        return;
    }

    results.textContent = "Analyzing text. Please wait...";
    analyzeButton.disabled = true;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Text analysis failed.");
        }

        results.replaceChildren();

        const heading = document.createElement("h3");
        heading.textContent = "Text Analysis Results";
        results.appendChild(heading);

        Object.entries(data).forEach(function (entry) {
            const key = entry[0];
            const value = entry[1];

            const paragraph = document.createElement("p");

            let displayValue;

            if (value !== null && typeof value === "object") {
                displayValue = JSON.stringify(value);
            } else {
                displayValue = String(value);
            }

            paragraph.textContent =
                key.replaceAll("_", " ") + ": " + displayValue;

            results.appendChild(paragraph);
        });

    } catch (error) {
        results.textContent = "Error: " + error.message;

    } finally {
        analyzeButton.disabled = false;
    }
});


// ===============================
// CHARACTER COUNTER
// ===============================

textInput.addEventListener("input", function () {
    charCount.textContent =
        textInput.value.length + " / 10000";
});


// ===============================
// IMAGE PREVIEW
// ===============================

imageInput.addEventListener("change", function () {
    imageResults.replaceChildren();

    const file = imageInput.files[0];

    if (!file) {
        imagePreviewContainer.hidden = true;
        return;
    }

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {
        imageInput.value = "";
        imagePreviewContainer.hidden = true;

        imageResults.textContent =
            "Please select a JPEG, PNG, or WebP image.";

        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        imageInput.value = "";
        imagePreviewContainer.hidden = true;

        imageResults.textContent =
            "The image must be smaller than 10 MB.";

        return;
    }

    imagePreview.src = URL.createObjectURL(file);
    imagePreviewContainer.hidden = false;

    imageDetails.textContent =
        file.name + " | " +
        (file.size / 1024 / 1024).toFixed(2) + " MB";
});


// ===============================
// IMAGE DETECTOR
// ===============================

analyzeImageButton.addEventListener("click", async function () {
    const file = imageInput.files[0];

    if (!file) {
        imageResults.textContent =
            "Please choose an image first.";

        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    imageResults.textContent =
        "Analyzing image. This may take a little while...";

    analyzeImageButton.disabled = true;

    try {
        const response = await fetch("/predict-image", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Image analysis failed."
            );
        }

        imageResults.replaceChildren();

        const heading = document.createElement("h3");
        heading.textContent = "Image Analysis Results";
        imageResults.appendChild(heading);

        const classification = document.createElement("p");

        classification.textContent =
            "Classification: " +
            String(data.classification || "unknown")
                .replaceAll("_", " ");

        imageResults.appendChild(classification);

        const probability = document.createElement("p");

        const score = Number(data.ai_probability);

        probability.textContent =
            "AI-generated probability: " +
            (Number.isFinite(score) ? (score * 100).toFixed(2) + "%" : "Unavailable");

        imageResults.appendChild(probability);

        const dimensions = document.createElement("p");

        dimensions.textContent =
            "Dimensions: " +
            data.image_width + " × " + data.image_height;

        imageResults.appendChild(dimensions);

        const format = document.createElement("p");
        format.textContent = "Format: " + data.image_format;

        imageResults.appendChild(format);

        const disclaimer = document.createElement("p");

        disclaimer.className = "disclaimer";

        disclaimer.textContent =
            data.disclaimer ||
            "This prediction is an estimate, not proof of image origin.";

        imageResults.appendChild(disclaimer);

    } catch (error) {
        imageResults.textContent =
            "Error: " + error.message;

    } finally {
        analyzeImageButton.disabled = false;
    }
});


// ===============================
// DARK / LIGHT MODE
// ===============================

function setTheme(dark) {
    document.body.classList.toggle("dark-mode", dark);

    themeIcon.textContent = dark ? "☀" : "☾";

    themeLabel.textContent =
        dark ? "Light mode" : "Dark mode";

    try {
        localStorage.setItem(
            "ai-shield-theme",
            dark ? "dark" : "light"
        );
    } catch (error) {
        // Theme still works if browser storage is unavailable.
    }
}

let savedTheme = "light";

try {
    savedTheme =
        localStorage.getItem("ai-shield-theme") || "light";
} catch (error) {
    // Use light mode by default.
}

setTheme(savedTheme === "dark");

themeToggle.addEventListener("click", function () {
    const dark =
        !document.body.classList.contains("dark-mode");

    setTheme(dark);
});


// ===============================
// INITIALIZE DASHBOARD
// ===============================

console.log("AI Shield dashboard JavaScript loaded successfully.");