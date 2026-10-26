from collections import Counter
from math import sqrt

from flask import Flask, jsonify, render_template, request
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

app = Flask(__name__)

tokenizer = RegexpTokenizer(r"[A-Za-z0-9']+")
stemmer = PorterStemmer()
stop_words = {
    "a", "an", "are", "can", "do", "does", "for", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "the", "to", "was", "what", "when",
    "where", "which", "who", "will", "with", "you", "your",
}

faqs = [
    {
        "question": "How long does shipping take?",
        "phrases": [
            "How long does shipping take?",
            "When will my order arrive?",
            "How many days for delivery?",
            "What is the delivery time?",
        ],
        "answer": "Standard delivery usually takes 3-5 business days. Delivery times can vary by location, and you can track your order for the latest update.",
    },
    {
        "question": "How much does shipping cost?",
        "phrases": [
            "How much does shipping cost?",
            "Is delivery free?",
            "What are the shipping fees?",
            "Do I have to pay for delivery?",
        ],
        "answer": "Standard shipping is free on orders over $50. For smaller orders, the shipping fee is shown at checkout before you place your order.",
    },
    {
        "question": "How can I track my order?",
        "phrases": [
            "How can I track my order?",
            "Where is my package?",
            "Can I get a delivery update?",
            "How do I check my shipment status?",
        ],
        "answer": "Open the order confirmation or shipping email and select its tracking link. You can also find the tracking number in the Orders section of your account.",
    },
    {
        "question": "Can I cancel or change an order?",
        "phrases": [
            "Can I cancel my order?",
            "How do I change an order?",
            "Can I edit my purchase?",
            "I need to stop my order",
        ],
        "answer": "Orders can usually be changed or cancelled within one hour of purchase. Visit the Orders section as soon as possible; shipped orders can no longer be changed.",
    },
    {
        "question": "What is your return policy?",
        "phrases": [
            "What is your return policy?",
            "How do I return an item?",
            "Can I send back my purchase?",
            "How many days do I have to return something?",
        ],
        "answer": "Most unused items can be returned within 30 days of delivery in their original packaging. Start a return from the Orders section to get return instructions.",
    },
    {
        "question": "When will I get my refund?",
        "phrases": [
            "When will I get my refund?",
            "How long does a refund take?",
            "Where is my money back?",
            "Has my return been refunded?",
        ],
        "answer": "After we receive and inspect a return, refunds are usually issued within 5-10 business days. Your bank may need additional time to show the credit.",
    },
    {
        "question": "Which payment methods do you accept?",
        "phrases": [
            "Which payment methods do you accept?",
            "Can I pay by credit card?",
            "Do you accept PayPal?",
            "What ways can I pay?",
        ],
        "answer": "We accept major credit and debit cards, PayPal, and supported digital wallets. Available options are displayed securely at checkout.",
    },
    {
        "question": "Is cash on delivery available?",
        "phrases": [
            "Is cash on delivery available?",
            "Can I pay when my order arrives?",
            "Do you accept cash at delivery?",
        ],
        "answer": "Cash on delivery is available in selected areas. Enter your address at checkout to see whether this payment option is available for your order.",
    },
    {
        "question": "How do I use a discount code?",
        "phrases": [
            "How do I use a discount code?",
            "Where do I enter a coupon?",
            "My promo code is not working",
            "How can I apply a voucher?",
        ],
        "answer": "Enter your discount code in the promo-code box at checkout, then select Apply. Check the code's expiry date and terms if it is not accepted.",
    },
    {
        "question": "Do I need an account to place an order?",
        "phrases": [
            "Do I need an account to place an order?",
            "Can I check out as a guest?",
            "How do I create an account?",
            "Can I shop without signing up?",
        ],
        "answer": "You can check out as a guest or create an account. An account makes it easier to track orders, save addresses, and view your purchase history.",
    },
    {
        "question": "What should I do if my item arrived damaged?",
        "phrases": [
            "What should I do if my item arrived damaged?",
            "My product is broken",
            "I received a damaged item",
            "The package arrived in bad condition",
        ],
        "answer": "Sorry about that. Contact our support team within 7 days of delivery with your order number and a photo of the damage so we can help.",
    },
    {
        "question": "Can I change my delivery address?",
        "phrases": [
            "Can I change my delivery address?",
            "How do I update my shipping address?",
            "I entered the wrong address",
            "Can you deliver to a different address?",
        ],
        "answer": "If the order has not shipped, contact support as soon as possible with the correct address. Once it has shipped, the address usually cannot be changed.",
    },
    {
        "question": "Do you ship internationally?",
        "phrases": [
            "Do you ship internationally?",
            "Can you deliver to another country?",
            "Where do you ship?",
            "Is international delivery available?",
        ],
        "answer": "International shipping is available to selected countries. Enter your delivery address at checkout to see available shipping options, costs, and estimated delivery times.",
    },
    {
        "question": "How can I contact customer support?",
        "phrases": [
            "How can I contact customer support?",
            "I want to talk to a person",
            "How do I reach customer service?",
            "Can I speak with a support agent?",
        ],
        "answer": "You can contact customer support using the Help or Contact Us link on our website. Include your order number so the team can assist you faster.",
    },
]


def preprocess(text):
    """Tokenize, remove common stop words, and stem tokens with NLTK."""
    return [
        stemmer.stem(token.lower())
        for token in tokenizer.tokenize(text)
        if token.lower() not in stop_words
    ]


def cosine_similarity(left, right):
    left_counts = Counter(left)
    right_counts = Counter(right)
    if not left_counts or not right_counts:
        return 0.0

    dot_product = sum(
        count * right_counts[token] for token, count in left_counts.items()
    )
    left_length = sqrt(sum(count * count for count in left_counts.values()))
    right_length = sqrt(sum(count * count for count in right_counts.values()))
    return dot_product / (left_length * right_length)


def find_best_faq(user_question):
    question_tokens = preprocess(user_question)
    best_faq = None
    best_score = 0.0

    for faq in faqs:
        for phrase in faq["phrases"]:
            score = cosine_similarity(question_tokens, preprocess(phrase))
            if score > best_score:
                best_faq = faq
                best_score = score

    return best_faq, best_score


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify(error="Send your question as JSON."), 400

    question = payload.get("message")
    if not isinstance(question, str) or not question.strip():
        return jsonify(error="Please type a question first."), 400
    if len(question) > 500:
        return jsonify(error="Please keep your question under 500 characters."), 400

    faq, score = find_best_faq(question.strip())
    if faq is None or score < 0.20:
        return jsonify(
            answer=(
                "I couldn't find a close match. Try asking about shipping, "
                "tracking, returns, refunds, payments, or your account."
            ),
            matched_question=None,
            score=round(score, 3),
        )

    return jsonify(
        answer=faq["answer"],
        matched_question=faq["question"],
        score=round(score, 3),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
