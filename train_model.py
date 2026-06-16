import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("Loading Dataset...")

df = pd.read_csv("dataset/mental_health_unbanlanced.csv")

# Input and Output
X = df["text"]
y = df["status"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=15000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000
        )
    )
])

print("Training Model...")

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy:.4f}")

# Save Model
joblib.dump(
    pipeline,
    "model/mental_health_model.pkl"
)

print("Model Saved Successfully")