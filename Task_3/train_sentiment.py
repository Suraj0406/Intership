import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# --------------------------------------------------
# 1. Setup
# --------------------------------------------------
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

DATA_PATH = r"G:\python\Task 3\twitter_training.csv"

# --------------------------------------------------
# 2. Load & Prepare Data
# --------------------------------------------------
def load_dataset(path):
    df = pd.read_csv(path, header=None, names=["tweet_id", "entity", "sentiment", "text"])
    df = df[df["sentiment"].isin(["Positive", "Negative", "Neutral"])]
    df["label"] = df["sentiment"].map({"Negative": 0, "Neutral": 1, "Positive": 2})
    return df.dropna(subset=["text"])

data = load_dataset(DATA_PATH)

# --------------------------------------------------
# 3. Text Cleaning
# --------------------------------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

data["clean_text"] = data["text"].apply(preprocess)

# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# --------------------------------------------------
# 5. Feature Extraction (TF-IDF)
# --------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------------------------------
# 6. Model Training
# --------------------------------------------------
classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
classifier.fit(X_train_vec, y_train)

# Save model
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(classifier, "logistic_model.pkl")

# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------
predictions = classifier.predict(X_test_vec)

print("\nModel Performance")
print("------------------")
print("Accuracy:", accuracy_score(y_test, predictions))
print("F1 Score (Macro):", f1_score(y_test, predictions, average="macro"))
print("\nDetailed Report:\n")
print(classification_report(y_test, predictions, target_names=["Negative", "Neutral", "Positive"]))

# --------------------------------------------------
# 8. Confusion Matrix Visualization
# --------------------------------------------------
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.title("Sentiment Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --------------------------------------------------
# 9. Test on Custom Sentences
# --------------------------------------------------
sample_texts = [
    "I love this product",
    "This is the worst thing ever",
    "It's okay, nothing special"
]

sample_clean = [preprocess(t) for t in sample_texts]
sample_vec = vectorizer.transform(sample_clean)
sample_preds = classifier.predict(sample_vec)

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

print("\nSample Predictions")
print("------------------")
for t, p in zip(sample_texts, sample_preds):
    print(f"{t} --> {label_map[p]}")

print("\nPROJECT COMPLETED SUCCESSFULLY WITH POSITIVE, NEGATIVE, AND NEUTRAL SENTIMENT")
