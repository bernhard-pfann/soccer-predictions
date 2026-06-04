from datetime import datetime

import pandas as pd

MIN_DATE = datetime(2015, 1, 1)


def _load_games(path: str) -> pd.DataFrame:
    cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    return pd.read_csv(path, usecols=cols)


def _clean_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] >= MIN_DATE]  # pyright: ignore[reportAssignmentType]
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def process_games(path: str) -> pd.DataFrame:
    df = _load_games(path)
    df = _clean_games(df)
    return df
