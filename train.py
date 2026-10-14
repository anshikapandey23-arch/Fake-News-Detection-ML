
import pandas as pd
import re
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------------
# 1. Load the datasets
# -----------------------------------------

print("Loading datasets...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0  # 0 = Fake News
true["label"] = 1  # 1 = Real News

# Combine fake and real news
data = pd.concat([fake, true], ignore_index=True)

# Handle missing values
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Combine title and article text
data["content"] = data["title"] + " " + data["text"]

data = data[["content", "label"]]

# Remove empty articles
data = data[data["content"].str.strip() != ""].copy()


# -----------------------------------------
# 2. Clean the text
# -----------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


print("Cleaning article text...")

data["content"] = data["content"].apply(clean_text)

# Remove exact duplicate articles after cleaning
data = data.drop_duplicates(subset=["content"]).reset_index(drop=True)

# Remove articles that became empty after cleaning
data = data[data["content"].str.strip() != ""].copy()

# Report articles whose identical content has conflicting labels
conflicting = data.groupby("content")["label"].nunique()
conflicting_count = int((conflicting > 1).sum())

print("Conflicting-label article groups:", conflicting_count)

# Remove any remaining exact-content groups with conflicting labels
conflict_texts = conflicting[conflicting > 1].index
data = data[~data["content"].isin(conflict_texts)].copy()
data = data.reset_index(drop=True)


# -----------------------------------------
# 3. Display dataset information
# -----------------------------------------

X = data["content"]
y = data["label"]

print("\nDataset Information")
print("-------------------")
print("Total articles:", len(data))
print("Fake articles:", int((y == 0).sum()))
print("Real articles:", int((y == 1).sum()))


# -----------------------------------------
# 4. Split the data before TF-IDF
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------------------
# 5. Convert text into TF-IDF features
# -----------------------------------------

print("\nConverting text into TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=3,
    max_features=50000,
    ngram_range=(1, 1),
    sublinear_tf=True
)

# Fit only on training data to avoid test-set leakage
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data using the already-fitted vectorizer
X_test_tfidf = vectorizer.transform(X_test)

print("Number of TF-IDF features:", X_train_tfidf.shape[1])


# -----------------------------------------
# 6. Train the Logistic Regression model
# -----------------------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# -----------------------------------------
# 7. Evaluate the model
# -----------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("====================================")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        digits=4,
        zero_division=0
    )
)


# -----------------------------------------
# 8. Save the model and vectorizer
# -----------------------------------------

print("\nSaving model files...")

Path("model").mkdir(parents=True, exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
print("Model path: model/model.pkl")
print("Vectorizer path: model/vectorizer.pkl")
print("\nTraining completed successfully!")
