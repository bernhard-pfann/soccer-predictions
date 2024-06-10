import pandas as pd
from datetime import datetime


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df[df["date"] > datetime(2015, 1, 1)]
    df["home_score"] = df["home_score"].astype(int, errors="ignore")
    df["away_score"] = df["away_score"].astype(int, errors="ignore")
    return df

def clean_rankings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(by=["country", "date"], as_index=False).agg({"rank": "mean"})
    df["rank"] = df.groupby("country")["rank"].ffill().astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def merge_rankings(df: pd.DataFrame, rankings: pd.DataFrame, side: str) -> pd.DataFrame:
    tmp = pd.merge_asof(
        left=df, 
        right=rankings, 
        left_on="date", 
        right_on="date", 
        left_by=side, 
        right_by="country", 
        direction="backward"
    )
    return (tmp
        .rename(columns={"rank": f"{side}_rank"})
        .drop(columns=["country"])
    )


def swap_names(lst: list, swap: tuple) -> list:
    return (lst
        .str.replace(swap[0], "$")
        .str.replace(swap[1], swap[0])
        .str.replace("$", swap[1])
    )


def duplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
    swapped_columns = swap_names(df.columns, swap=("home", "away"))
    swapped_df = df.copy()
    swapped_df.columns = swapped_columns
    
    combined_df = pd.concat([df, swapped_df], ignore_index=True)
    combined_df["rank_diff"] = (combined_df["home_team_rank"] - combined_df["away_team_rank"])
    combined_df.sort_values(by="date", inplace=True)

    return combined_df