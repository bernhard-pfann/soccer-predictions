import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm, expon


def normalize(series: pd.Series) -> pd.Series:
    return series / series.sum()


def get_ranking(df: pd.DataFrame, team: str, day: datetime) -> int:
    return df[
        (df["country"] == team) & 
        (df["date"] <= day)
    ].iloc[-1]["rank"]


def get_comparable_matches(
    df: pd.DataFrame, 
    team: str,
    day: datetime, 
    rank_diff: int, 
    time_sigma: float, 
    rank_sigma: float
) -> pd.DataFrame:
    matches = df[df["home_team"]==team].copy()
    matches["days_diff"] = (day - matches["date"]).dt.days
    matches["time_weight"] = normalize(expon.pdf(matches["days_diff"], scale=time_sigma))
    matches["rank_weight"] = normalize(norm.pdf(matches["rank_diff"], loc=rank_diff, scale=rank_sigma))
    matches["weight"] = normalize(np.sqrt(matches["time_weight"] * matches["rank_weight"]))
    matches.sort_values(by="weight", ascending=False, inplace=True)
    return matches


def get_offense_defense(df: pd.DataFrame) -> tuple[float, float]:
    offense = (df["home_score"] * df["weight"]).sum()
    defense = (df["away_score"] * df["weight"]).sum()
    return offense, defense


def draw_result(
    home_stats: tuple[float, float], 
    away_stats: tuple[float, float], 
    n: int=1
) -> tuple[int, int] | tuple[list[int], list[int]]:
    
    home_lambda = (home_stats[0] + away_stats[1]) / 2
    away_lambda = (home_stats[1] + away_stats[0]) / 2
    home_samples = np.random.poisson(home_lambda, n)
    away_samples = np.random.poisson(away_lambda, n)
    
    if n == 1:
        return home_samples.item(), away_samples.item()
    return home_samples.tolist(), away_samples.tolist()


def predict_score(
    df: pd.DataFrame, 
    rankings: pd.DataFrame, 
    home_team: str, 
    away_team: str, 
    day: datetime, 
    time_sigma: float=90, 
    rank_sigma: float=15
) -> tuple[int, int]:
    
    rank_diff = (
        get_ranking(df=rankings, team=home_team, day=day) - 
        get_ranking(df=rankings, team=away_team, day=day)
    )
    args = {
        "day": day, 
        "time_sigma": time_sigma, 
        "rank_sigma": rank_sigma
    }
    home_team_matches = get_comparable_matches(df=df, team=home_team, rank_diff=rank_diff, **args)
    away_team_matches = get_comparable_matches(df=df, team=away_team, rank_diff=-rank_diff, **args)
    home_team_stats = get_offense_defense(home_team_matches)
    away_team_stats = get_offense_defense(away_team_matches)
    
    return draw_result(
        home_stats=home_team_stats, 
        away_stats=away_team_stats
    )
