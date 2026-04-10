import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

import joblib

df = pd.read_csv("../../Datasets/tmdb_5000_movies.csv")

df.dropna(inplace=True)

df["profit"] = df["revenue"] - df["budget"]
df["is_successful"] = df["profit"] > 0

X = df[
    [
        "budget",
        "popularity",
        "runtime",
    ]
]

y = df["is_successful"]

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

# Old manual preprocessing (kept for reference):
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)

model = LogisticRegression()

# Old manual training/prediction (kept for reference):
# model.fit(X_train_scaled, y_train)
# y_pred = model.predict(X_test_scaled)

# Build a single pipeline to apply scaling + logistic regression.
model_pipeline = Pipeline(
    [
        ("scaler", scaler),  # Standardize numeric features
        ("classifier", model),  # Train LogisticRegression on scaled data
    ]
)

# Train and predict using the pipeline (same features/model settings).
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

# print(model.score(X_test_scaled, y_test))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

joblib.dump(model_pipeline, "models/movie_success_classification_model.pkl")
# joblib.dump(
#     model_pipeline.named_steps["scaler"], "models/movie_success_classification_scaler.pkl"
# )
