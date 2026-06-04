from datetime import datetime

import pandas as pd

MIN_DATE = datetime(2015, 1, 1)


def load_games(path: str) -> pd.DataFrame:
    cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    return pd.read_csv(path, usecols=cols)


def clean_games(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] >= MIN_DATE]  # pyright: ignore[reportAssignmentType]
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def process_games(path: str) -> pd.DataFrame:
    df = load_games(path)
    df = clean_games(df)
    return df
