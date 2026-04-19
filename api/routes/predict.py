import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class Movie(BaseModel):
    budget: float
    popularity: float
    runtime: float
    original_language: str
    genres: str
    keywords: str
    production_companies: str
    production_countries: str


model = joblib.load("models/movie_success_pipeline.pkl")


@app.post("/predict")
def predict_movie_success(movie: Movie):
    input_data = [
        [
            movie.budget,
            movie.popularity,
            movie.runtime,
            movie.original_language,
            movie.genres,
            movie.keywords,
            movie.production_companies,
            movie.production_countries,
        ]
    ]

    input_df = pd.DataFrame(
        input_data,
        columns=[
            "budget",
            "popularity",
            "runtime",
            "original_language",
            "genres",
            "keywords",
            "production_companies",
            "production_countries",
        ],
    )

    prediction = model.predict(input_df)

    return {"is_successful": bool(prediction[0])}
