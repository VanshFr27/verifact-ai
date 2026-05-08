from flask import Flask, render_template, request
import pickle
import json
import os
import re
import sqlite3

app = Flask(__name__)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ------------------ LOAD KNOWLEDGE BASE ------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "knowledge_base.json")

with open(file_path, "r", encoding="utf-8") as f:
    known_facts = json.load(f)

# ------------------ CLEAN TEXT ------------------
def clean_text(text):
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z ]', '', text)

    # Remove unnecessary words
    stop_words = {
        "is", "am", "are", "was", "were",
        "do", "does", "did",
        "the", "a", "an",
        "of", "to", "in", "on",
        "for", "with", "at",
        "can", "could", "should",
        "would", "will",
        "please"
    }

    words = text.split()

    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)

def check_contradiction(user_input, fact):
    opposite_words = {
        "healthy": "unhealthy",
        "unhealthy": "healthy",
        "real": "fake",
        "fake": "real",
        "good": "bad",
        "bad": "good",
        "safe": "dangerous",
        "dangerous": "safe",
        "true": "false",
        "false": "true"
    }

    user_words = set(user_input.split())
    fact_words = set(fact.split())

    for word in opposite_words:
        if word in user_words and opposite_words[word] in fact_words:
            return True

    return False

# ------------------ DATABASE SETUP ------------------
conn = sqlite3.connect("history.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS history (input TEXT, result TEXT)")
conn.commit()
conn.close()

# ------------------ MAIN ROUTE ------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        # User input
        user_input = clean_text(request.form["news"])

        # Exact match
        if user_input in known_facts:

            prediction = known_facts[user_input]

        else:

            # Smart matching
            matched = False

            for fact in known_facts:

                keywords = fact.split()

                match_count = sum(
                    1 for word in keywords if word in user_input
                )

                if match_count >= max(2, len(keywords)//2):

                    # Contradiction check
                    if check_contradiction(user_input, fact):

                        if "Real News" in known_facts[fact]:
                            prediction = "Fake News ❌"

                        else:
                            prediction = "Real News ✅"

                    else:
                        prediction = known_facts[fact]

                    matched = True
                    break

            # ML fallback
            if not matched:

                data = vectorizer.transform([user_input])

                result = model.predict(data)[0]

                prediction = (
                    "Real News ✅"
                    if result == 1
                    else "Fake News ❌"
                )

        # Save to database
        conn = sqlite3.connect("history.db")

        c = conn.cursor()

        c.execute(
            "INSERT INTO history VALUES (?, ?)",
            (user_input, prediction)
        )

        conn.commit()
        conn.close()

    return render_template(
        "index.html",
        prediction=prediction
    )

# ------------------ HISTORY API ------------------
@app.route("/history")
def history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT input, result FROM history ORDER BY ROWID DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return {"history": rows}

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
