from typing import Literal

import pandas as pd


def attach_rankings(df: pd.DataFrame, rankings: pd.DataFrame, side: Literal["home_team", "away_team"]) -> pd.DataFrame:
    tmp = pd.merge_asof(
        left=df,
        right=rankings,
        left_on="date",
        right_on="date",
        left_by=side,
        right_by="country",
        direction="backward",
    )
    return tmp.rename(columns={"rank": f"{side}_rank"}).drop(columns=["country"])


def enrich_results(results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    df = attach_rankings(results, rankings, "home_team")
    df = attach_rankings(df, rankings, "away_team")
    df = df.dropna(subset=["home_team_rank", "away_team_rank"])
    
    return df.astype({
        "home_team_rank": int,
        "away_team_rank": int,
    })


def swap_names(columns: pd.Index, swap: tuple[str, str]) -> pd.Index:
    return (
        columns.str.replace(swap[0], "__tmp__")
        .str.replace(swap[1], swap[0])
        .str.replace("__tmp__", swap[1])
    )


def duplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
    swapped_columns = swap_names(df.columns, swap=("home", "away"))  # pyright: ignore[reportArgumentType]
    swapped_df = df.copy()
    swapped_df.columns = swapped_columns

    combined_df = pd.concat([df, swapped_df], ignore_index=True)
    combined_df["rank_diff"] = combined_df["home_team_rank"] - combined_df["away_team_rank"]
    combined_df.sort_values(by="date", inplace=True)
    return combined_df


def build_dataset(results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    df = enrich_results(results, rankings)
    df = duplicate_matches(df)
    df["score"] = list(zip(df["home_score"], df["away_score"]))
    return df