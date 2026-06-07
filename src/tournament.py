import random
from collections import defaultdict
from datetime import datetime

import pandas as pd

from src.predict import MatchPredictor


class Tournament:
    FIXTURE_COUNT_TO_ROUND = {
        16: "round_of_32",
        8: "round_of_16",
        4: "quarter_final",
        2: "semi_final",
        1: "final",
    }

    def __init__(self, predictor: MatchPredictor, match_day: datetime) -> None:
        self.predictor = predictor
        self.match_day = match_day

        self.group_matches: pd.DataFrame | None = None
        self.group_tables: dict[str, pd.DataFrame] = {}

        self.knockout_games: dict[str, pd.DataFrame] = {}
        self.knockout_results: dict[str, pd.DataFrame] = {}
        self.champion: str | None = None

    def run(self, gameplan: pd.DataFrame) -> str:
        self.group_matches = self.predictor.simulate_batch(games=gameplan, day=self.match_day)
        self.group_matches["group"] = gameplan["group"].values

        self.group_tables = self._calculate_group_tables(self.group_matches)
        fixtures = self._build_round_of_32()

        while len(fixtures) > 0:
            round_name = self.FIXTURE_COUNT_TO_ROUND[len(fixtures)]

            # self.knockout_games[round_name] = fixtures

            predictions = self.predictor.simulate_batch(games=fixtures, day=self.match_day)
            predictions["winning_team"] = predictions.apply(self._choose_winner, axis=1)
            
            self.knockout_results[round_name] = predictions
            winners = predictions["winning_team"].tolist()

            if len(winners) == 1:
                self.champion = winners[0]
                return self.champion

            fixtures = self._build_next_round(winners)
    
    def _calculate_group_tables(self, matches: pd.DataFrame) -> dict[str, pd.DataFrame]:
        standings = defaultdict(lambda: {"punkte": 0, "tore": 0, "gegentore": 0})
        team_to_group = {}

        for row in matches.itertuples(index=False):
            home_goals, away_goals = row.score

            team_to_group[row.home_team] = row.group
            team_to_group[row.away_team] = row.group

            standings[row.home_team]["tore"] += home_goals
            standings[row.home_team]["gegentore"] += away_goals

            standings[row.away_team]["tore"] += away_goals
            standings[row.away_team]["gegentore"] += home_goals

            if home_goals > away_goals:
                standings[row.home_team]["punkte"] += 3
            elif away_goals > home_goals:
                standings[row.away_team]["punkte"] += 3
            else:
                standings[row.home_team]["punkte"] += 1
                standings[row.away_team]["punkte"] += 1

        table = pd.DataFrame([
            {
                "group": team_to_group[team],
                "team": team,
                **stats,
                "tordifferenz": (stats["tore"] - stats["gegentore"]),
            }
            for team, stats in standings.items()
        ])

        return {
            group: (table
                .query("group == @group")
                .sort_values(["punkte", "tordifferenz", "tore"], ascending=False)
                .reset_index(drop=True)
            )
            for group in sorted(table["group"].unique())
        }
    
    def _get_direct_qualifiers(self) -> dict[str, str]:
        qualified = {}

        for group, table in self.group_tables.items():
            qualified[f"{group}1"] = table.iloc[0]["team"]
            qualified[f"{group}2"] = table.iloc[1]["team"]

        return qualified

    def _get_best_third_places(self, n: int) -> list[str]:
        thirds = pd.concat(
            [table.iloc[[2]] for table in self.group_tables.values()],
            ignore_index=True,
        )
        return (thirds
            .sort_values(["punkte", "tordifferenz", "tore"], ascending=False)
            .head(n)["team"]
            .tolist()
        )
    
    def _build_round_of_32(self) -> pd.DataFrame:
        qualified = self._get_direct_qualifiers()
        best_thirds = self._get_best_third_places(n=8)
        
        matches = [
            (qualified["A1"], best_thirds[7]),
            (qualified["B1"], best_thirds[6]),
            (qualified["C1"], best_thirds[5]),
            (qualified["D1"], best_thirds[4]),
            (qualified["E1"], best_thirds[3]),
            (qualified["F1"], best_thirds[2]),
            (qualified["G1"], best_thirds[1]),
            (qualified["H1"], best_thirds[0]),
            (qualified["I1"], qualified["L2"]),
            (qualified["J1"], qualified["K2"]),
            (qualified["K1"], qualified["J2"]),
            (qualified["L1"], qualified["I2"]),
            (qualified["A2"], qualified["H2"]),
            (qualified["B2"], qualified["G2"]),
            (qualified["C2"], qualified["F2"]),
            (qualified["D2"], qualified["E2"]),
        ]
        return pd.DataFrame(matches, columns=["home_team", "away_team"])
    
    @staticmethod
    def _choose_winner(row: pd.Series) -> str:
        home_goals, away_goals = row["score"]

        if home_goals == away_goals:
            return random.choice([row["home_team"], row["away_team"]])

        return row["home_team"] if home_goals > away_goals else row["away_team"]

    @staticmethod
    def _build_next_round(winners: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            data=[(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)],
            columns=["home_team", "away_team"],
        )
    