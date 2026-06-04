from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import expon, norm


@dataclass(frozen=True)
class SimilarityConfig:
    """Configuration for match similarity weighting."""
    time_sigma: float = 90
    rank_sigma: float = 15


@dataclass(frozen=True)
class TeamStats:
    """Weighted offensive and defensive strength of a team."""
    offense: float
    defense: float


@dataclass(frozen=True)
class GameStats:
    """Expected goals for both teams of a game."""
    home: float
    away: float


@dataclass(frozen=True)
class PredictionResult:
    """Full prediction output including intermediate artifacts."""
    score: tuple[int, int] | tuple[list[int], list[int]]
    rank_diff: int
    home_matches: pd.DataFrame
    away_matches: pd.DataFrame
    home_stats: TeamStats
    away_stats: TeamStats
    game_stats: GameStats


class MatchPredictor:
    """Football match predictor based on weighted historical matches."""

    def __init__(self, matches: pd.DataFrame, rankings: pd.DataFrame, config: SimilarityConfig = SimilarityConfig()) -> None:
        self.matches = matches
        self.rankings = rankings
        self.config = config

    def predict(self, home_team: str, away_team: str, day: datetime, sample: bool = True, sample_size: int = 1) -> PredictionResult:
        """
        Predict a match and return all intermediate artifacts.

        :param home_team: Home team.
        :param away_team: Away team.
        :param day: Match date.
        :param sample: Whether to sample from Poisson distributions.
        :param sample_size: Number of samples to draw. This parameter is unused when sample = False.
        :return: Prediction result.
        """
        rank_diff = self._get_rank_difference(home_team=home_team, away_team=away_team, day=day)
        home_matches = self._get_comparable_matches(team=home_team, day=day, rank_diff=rank_diff)
        away_matches = self._get_comparable_matches(team=away_team, day=day, rank_diff=-rank_diff)

        home_stats = self._get_team_stats(home_matches)
        away_stats = self._get_team_stats(away_matches)
        game_stats = self._get_game_stats(home_stats=home_stats, away_stats=away_stats)

        if sample:
            score = self._draw_result(game_stats, n=sample_size)
        else:
            score = self._round_game_stats(game_stats)

        return PredictionResult(
            score=score,
            rank_diff=rank_diff,
            home_matches=home_matches,
            away_matches=away_matches,
            home_stats=home_stats,
            away_stats=away_stats,
            game_stats=game_stats,
        )

    @staticmethod
    def _normalize(series: pd.Series | np.ndarray) -> np.ndarray:
        series = np.asarray(series, dtype=np.float64)
        return series / series.sum()

    def _get_ranking(self, team: str, day: datetime) -> int:
        return int(
            self.rankings[
                (self.rankings["country"] == team) & 
                (self.rankings["date"] <= day)
            ]
            .iloc[-1]["rank"]
        )

    def _get_rank_difference(self, home_team: str, away_team: str, day: datetime) -> int:
        return (
            self._get_ranking(home_team, day) - 
            self._get_ranking(away_team, day)
        )

    def _get_comparable_matches(self, team: str, day: datetime, rank_diff: int) -> pd.DataFrame:
        df = self.matches[self.matches["home_team"] == team].copy()

        df["days_diff"] = (day - df["date"]).dt.days
        df["time_weight"] = self._normalize(expon.pdf(df["days_diff"], scale=self.config.time_sigma))
        df["rank_weight"] = self._normalize(norm.pdf(df["rank_diff"], loc=rank_diff, scale=self.config.rank_sigma))
        df["weight"] = self._normalize(np.sqrt(df["time_weight"] * df["rank_weight"]))
        return df.sort_values(by="weight", ascending=False)

    @staticmethod
    def _get_team_stats(matches: pd.DataFrame) -> TeamStats:
        return TeamStats(
            offense=(matches["home_score"] * matches["weight"]).sum(),
            defense=(matches["away_score"] * matches["weight"]).sum(),
        )

    @staticmethod
    def _get_game_stats(home_stats: TeamStats, away_stats: TeamStats) -> GameStats:
        return GameStats(
            home=(home_stats.offense + away_stats.defense) / 2,
            away=(away_stats.offense + home_stats.defense) / 2,
        )

    @staticmethod
    def _round_game_stats(game_stats: GameStats) -> tuple[int, int]:
        return (
            round(game_stats.home),
            round(game_stats.away),
        )

    @staticmethod
    def _draw_result(game_stats: GameStats, n: int = 1) -> tuple[int, int] | tuple[list[int], list[int]]:
        home_scores = np.random.poisson(game_stats.home, n)
        away_scores = np.random.poisson(game_stats.away, n)
        
        if n == 1:
            return home_scores.item(), away_scores.item()
        return home_scores.tolist(), away_scores.tolist()
