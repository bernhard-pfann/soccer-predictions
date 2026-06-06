from collections import Counter
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


@dataclass(frozen=True)
class SimulationResult:
    """Collection of simulated scorelines."""
    home_scores: list[int]
    away_scores: list[int]

    @property
    def scorelines(self) -> list[tuple[int, int]]:
        return list(zip(
            self.home_scores,
            self.away_scores,
        ))

    @property
    def most_likely_score(self) -> tuple[int, int]:
        return Counter(self.scorelines).most_common(1)[0][0]


class MatchPredictor:
    """Football match predictor based on weighted historical matches."""

    def __init__(self, dataset: pd.DataFrame, rankings: pd.DataFrame, config: SimilarityConfig = SimilarityConfig()) -> None:
        self.dataset = dataset
        self.rankings = rankings
        self.config = config

    def predict(self, home_team: str, away_team: str, day: datetime) -> PredictionResult:
        """
        Predict a match and return all intermediate artifacts.

        :param home_team: Home team.
        :param away_team: Away team.
        :param day: Match date.
        :return: Prediction result.
        """
        rank_diff = self._get_rank_difference(home_team=home_team, away_team=away_team, day=day)
        home_matches = self._get_comparable_matches(team=home_team, day=day, rank_diff=rank_diff)
        away_matches = self._get_comparable_matches(team=away_team, day=day, rank_diff=-rank_diff)

        home_stats = self._get_team_stats(home_matches)
        away_stats = self._get_team_stats(away_matches)
        game_stats = self._get_game_stats(home_stats=home_stats, away_stats=away_stats)
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

    def simulate(self, home_team: str, away_team: str, day: datetime, n_simulations: int = 10_000) -> SimulationResult:
        """
        Generate simulated scorelines for a predicted match.

        :param home_team: Name of the home team.
        :param away_team: Name of the away team.
        :param day: Date of the match.
        :param n_simulations: Number of simulated scorelines to generate.
        :return: SimulationResult containing sampled home and away scores.
        """
        prediction = self.predict(home_team=home_team, away_team=away_team, day=day)
        home_scores = np.random.poisson(prediction.game_stats.home, n_simulations)
        away_scores = np.random.poisson(prediction.game_stats.away, n_simulations)

        return SimulationResult(
            home_scores=home_scores.tolist(),
            away_scores=away_scores.tolist(),
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
        df = self.dataset[self.dataset["home_team"] == team].copy()

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
