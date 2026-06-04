import pandas as pd

COUNTRY_MAP = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Congo": "DR Congo",
}

def load_rankings(path: str) -> pd.DataFrame:
    cols = ["rank_date", "country_full", "rank"]
    df = pd.read_csv(path, parse_dates=["rank_date"], usecols=cols)
    df.rename(columns={"rank_date": "date", "country_full": "country"}, inplace=True)
    return df


def map_ranking_countries(df: pd.DataFrame) -> pd.DataFrame:
    df["country"] = df["country"].replace(COUNTRY_MAP)
    return df


def clean_rankings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(by=["country", "date"], as_index=False).agg({"rank": "mean"})
    df = df.assign(rank=df.groupby("country")["rank"].ffill().astype(int))
    df = df.sort_values("date").reset_index(drop=True)
    return df


def process_rankings(path: str) -> pd.DataFrame:
    df = load_rankings(path)
    df = map_ranking_countries(df)
    df = clean_rankings(df)
    return df
