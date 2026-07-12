from flask import Flask, render_template, request
import joblib
import re

# ===========================
# Create Flask App
# ===========================

app = Flask(__name__)

# ===========================
# Load Model and Vectorizer
# ===========================

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# ===========================
# Text Cleaning Function
# ===========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ===========================
# Home Page
# ===========================

@app.route("/")
def home():

    return render_template("index.html")


# ===========================
# Prediction Route
# ===========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        news = request.form["news"]

        cleaned_news = clean_text(news)

        vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(vector)

        probability = model.predict_proba(vector)

        confidence = max(probability[0]) * 100

        confidence = f"{confidence:.2f}%"

        if prediction[0] == 1:

            result = "✅ REAL NEWS"

            card_color = "success"

        else:

            result = "❌ FAKE NEWS"

            card_color = "danger"

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            card_color=card_color
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction="Something went wrong!",
            confidence="0%",
            card_color="warning"
        )


# ===========================
# Run App
# ===========================

if __name__ == "__main__":

    app.run(debug=True)