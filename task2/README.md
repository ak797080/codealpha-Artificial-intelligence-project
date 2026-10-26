# Online Shopping FAQ Chatbot

A small Flask chat app that answers online-shopping questions by finding the
closest FAQ with cosine similarity. NLTK tokenizes, filters common words, and
stems the question text before matching it against the FAQ phrases.

## Run on Windows

Open PowerShell in this `task2` folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000> in your browser. Keep the terminal window open
while using the chatbot. This app is served by Flask; use the Flask URL rather
than opening the HTML file with Live Server.

The FAQ data and match threshold are in `app.py`. No NLTK corpus download is
needed.
