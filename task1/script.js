const inputText = document.getElementById("inputText");
const outputText = document.getElementById("outputText");
const sourceLanguage = document.getElementById("sourceLanguage");
const targetLanguage = document.getElementById("targetLanguage");
const translateButton = document.getElementById("translateButton");
const copyButton = document.getElementById("copyButton");
const speakButton = document.getElementById("speakButton");
const swapButton = document.getElementById("swapButton");
const status = document.getElementById("status");

translateButton.addEventListener("click", translateText);

copyButton.addEventListener("click", async () => {
  if (!outputText.value) return;

  await navigator.clipboard.writeText(outputText.value);
  status.style.color = "#15803d";
  status.textContent = "Copied to clipboard!";
});

speakButton.addEventListener("click", () => {
  if (!outputText.value) return;

  const speech = new SpeechSynthesisUtterance(outputText.value);
  speech.lang = targetLanguage.value;
  window.speechSynthesis.speak(speech);
});

swapButton.addEventListener("click", () => {
  if (sourceLanguage.value === "auto") {
    status.style.color = "#b91c1c";
    status.textContent = "Auto Detect cannot be swapped.";
    return;
  }

  const temp = sourceLanguage.value;
  sourceLanguage.value = targetLanguage.value;
  targetLanguage.value = temp;
});

async function translateText() {
  const text = inputText.value.trim();

  if (!text) {
    status.style.color = "#b91c1c";
    status.textContent = "Please enter some text.";
    return;
  }

  if (text.length > 5000) {
    status.style.color = "#b91c1c";
    status.textContent = "Please enter fewer than 5000 characters.";
    return;
  }

  status.style.color = "#374151";
  status.textContent = "Translating...";
  outputText.value = "";

  try {
    const params = new URLSearchParams({
      client: "gtx",
      sl: sourceLanguage.value === "auto" ? "auto" : sourceLanguage.value,
      tl: targetLanguage.value,
      dt: "t",
      q: text
    });
    const response = await fetch(
      `https://translate.googleapis.com/translate_a/single?${params}`
    );

    if (!response.ok) {
      throw new Error("Translation service is unavailable. Please try again.");
    }

    const data = await response.json();
    const translatedText = Array.isArray(data[0])
      ? data[0].map((segment) => segment[0] || "").join("")
      : "";

    if (!translatedText) {
      throw new Error("No translation was returned. Please try again.");
    }

    outputText.value = translatedText;
    status.style.color = "#15803d";
    status.textContent = "Translation completed.";
  } catch (error) {
    status.style.color = "#b91c1c";
    status.textContent = error.message || "Translation failed. Please try again.";
  }
}