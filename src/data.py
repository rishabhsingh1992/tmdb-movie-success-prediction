import ast
import json
from pathlib import Path

import pandas as pd


def parse_name_list(raw_payload: str) -> str:
    if not isinstance(raw_payload, str) or not raw_payload.strip():
        return ""

    payload = raw_payload.strip()
    parsed = None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(payload)
        except (ValueError, SyntaxError):
            return ""

    if not isinstance(parsed, list):
        return ""

    names = [item.get("name", "") for item in parsed if isinstance(item, dict)]
    names = [
        name.strip().lower() for name in names if isinstance(name, str) and name.strip()
    ]
    return " ".join(names)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["profit"] = df["revenue"] - df["budget"]
    df["genres"] = df["genres"].apply(parse_name_list)
    df["keywords"] = df["keywords"].apply(parse_name_list)
    df["production_companies"] = df["production_companies"].apply(parse_name_list)
    df["production_countries"] = df["production_countries"].apply(parse_name_list)

    df = df.dropna(
        subset=[
            "budget",
            "popularity",
            "runtime",
            "original_language",
            "genres",
            "keywords",
            "production_companies",
            "production_countries",
            "profit",
        ]
    )

    return df
