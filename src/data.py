import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df
