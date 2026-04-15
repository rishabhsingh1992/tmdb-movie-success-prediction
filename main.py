from src.data import load_data
from src.features import build_features
from src.model import evaluate_model, save_model, split_data, train_model

df = load_data("../../Datasets/tmdb_5000_movies.csv")

X, y = build_features(df)

X_train, X_test, y_train, y_test = split_data(X, y)

pipeline = train_model(X_train, y_train)

evaluate_model(pipeline, X_test, y_test)

save_model(pipeline, "models/movie_success_classification_model_pipeline.pkl")
