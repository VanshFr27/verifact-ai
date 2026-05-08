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
    text = re.sub(r'[^a-z ]', '', text)
    return text.strip()

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
        user_input = clean_text(request.form["news"])

        # 🔥 1. Exact Match (highest priority)
        if user_input in known_facts:
            prediction = known_facts[user_input]

        else:
            # 🔹 2. Smart Keyword Matching
            matched = False
            for fact in known_facts:
                keywords = fact.split()
                match_count = sum(1 for word in keywords if word in user_input)

                if match_count >= 2:
                    prediction = known_facts[fact]
                    matched = True
                    break

            # 🔹 3. ML Model Fallback
            if not matched:
                data = vectorizer.transform([user_input])
                result = model.predict(data)[0]

                proba = model.predict_proba(data)[0]
                confidence = max(proba) * 100

                prediction = ("Real News ✅" if result == 1 else "Fake News ❌") + f" ({confidence:.2f}% sure)"

        # 🔹 Save to database
        conn = sqlite3.connect("history.db")
        c = conn.cursor()
        c.execute("INSERT INTO history VALUES (?, ?)", (user_input, prediction))
        conn.commit()
        conn.close()

    return render_template("index.html", prediction=prediction)

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