import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# Load datasets
# -----------------------------------
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# -----------------------------------
# Add labels
# Fake = 0
# Real = 1
# -----------------------------------
fake["label"] = 0
true["label"] = 1

# -----------------------------------
# Merge datasets
# -----------------------------------
data = pd.concat([fake, true], ignore_index=True)

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

# -----------------------------------
# Combine Title + Text
# -----------------------------------
data["content"] = data["title"] + " " + data["text"]

# Keep only required columns
data = data[["content", "label"]]

# -----------------------------------
# Text Cleaning
# -----------------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

data["content"] = data["content"].apply(clean_text)

# -----------------------------------
# Features & Labels
# -----------------------------------
X = data["content"]

y = data["label"]

# -----------------------------------
# TF-IDF Vectorization
# -----------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X = vectorizer.fit_transform(X)

print("Total Features :", X.shape)

# -----------------------------------
# Split Dataset
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# -----------------------------------
# Machine Learning Model
# -----------------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")

# -----------------------------------
# Prediction
# -----------------------------------
y_pred = model.predict(X_test)

# -----------------------------------
# Accuracy
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy : {accuracy*100:.2f}%")

# -----------------------------------
# Confusion Matrix
# -----------------------------------
print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# Classification Report
# -----------------------------------
print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# -----------------------------------
# Save Model
# -----------------------------------
joblib.dump(model, "model/model.pkl")

joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel Saved Successfully!")