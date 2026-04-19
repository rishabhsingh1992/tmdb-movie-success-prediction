from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLS = ["budget", "popularity", "runtime"]
CATEGORICAL_COLS = ["original_language"]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("genres", TfidfVectorizer(), "genres"),
            ("keywords", TfidfVectorizer(), "keywords"),
            ("production_companies", TfidfVectorizer(), "production_companies"),
            ("production_countries", TfidfVectorizer(), "production_countries"),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression()),
        ]
    )


def train_and_evaluate(X, y) -> Pipeline:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    return pipeline


def save_model(pipeline: Pipeline, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
