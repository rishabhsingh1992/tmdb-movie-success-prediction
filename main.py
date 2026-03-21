import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = LogisticRegression()

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# print(model.score(X_test_scaled, y_test))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

joblib.dump(model, "models/movie_success_classification_model.pkl")
joblib.dump(scaler, "models/movie_success_classification_scaler.pkl")
