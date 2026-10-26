const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const messages = document.getElementById("messages");
const suggestionButtons = document.querySelectorAll(".suggestion");

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendQuestion(questionInput.value);
});

suggestionButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    await sendQuestion(button.dataset.question);
  });
});

async function sendQuestion(question) {
  const message = question.trim();
  if (!message || sendButton.disabled) return;

  addMessage(message, "user");
  questionInput.value = "";
  sendButton.disabled = true;
  questionInput.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "The question could not be sent.");
    }

    addMessage(data.answer, "bot", data.matched_question);
  } catch (error) {
    addMessage(
      error.message || "Could not reach the chatbot. Please try again.",
      "bot",
    );
  } finally {
    sendButton.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

function addMessage(text, sender, matchedQuestion) {
  const article = document.createElement("article");
  article.className = `message ${sender}-message`;

  if (sender === "bot") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "S";
    article.append(avatar);
  }

  const content = document.createElement("div");
  content.className = "message-content";

  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = sender === "user" ? "YOU" : "SHOPWISE ASSISTANT";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  content.append(label, bubble);

  if (matchedQuestion) {
    const matchNote = document.createElement("p");
    matchNote.className = "match-note";
    matchNote.textContent = `Matched FAQ: ${matchedQuestion}`;
    content.append(matchNote);
  }

  article.append(content);
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
}
