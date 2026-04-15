def build_features(df):
    df["profit"] = df["revenue"] - df["budget"]
    df["is_successful"] = df["profit"] > 0

    X = df[["budget", "popularity", "runtime"]]
    y = df["is_successful"]

    return X, y
