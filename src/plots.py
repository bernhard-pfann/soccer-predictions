import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_comparable_matches(matches: pd.DataFrame, team: str, top_n: int = 10, weight_col: str = "weight") -> None:
    """Plot the most comparable matches for a given team."""
    df = matches.sort_values(by="weight", ascending=False).head(top_n).copy()
    labels = (df["away_team"] + " (" + df["date"].dt.strftime("%Y-%m-%d") + ")")

    _, ax = plt.subplots(figsize=(10, 5), dpi=200)
    bars = ax.barh(y=labels, width=df[weight_col])
    
    max_weight = df[weight_col].max()
    margin = max_weight * 0.01
    ax.set_xlim(0, max_weight * 1.15)

    for bar, score in zip(bars, df["score"]):
        ax.text(
            x=bar.get_width() + margin,
            y=bar.get_y() + bar.get_height() / 2,
            s=str(score),
            va="center",
            ha="left",
        )

    
    ax.invert_yaxis()  # highest weight on top
    ax.set_title(f"Comparable Matches for {team}")
    ax.set_xlabel(weight_col.replace("_", " ").title())
    plt.tight_layout()
    plt.show()


def plot_score_distribution(
    home_team: str,
    away_team: str,
    home_scores: list[int],
    away_scores: list[int],
) -> None:
    """Plot a heatmap of simulated score probabilities."""
    df = pd.crosstab(
        index=home_scores,
        columns=away_scores,
        rownames=[f"{home_team} Goals"],
        colnames=[f"{away_team} Goals"],
    ).sort_index(ascending=False)

    df = df.div(df.to_numpy().sum())

    plt.figure(figsize=(10, 10), dpi=200)
    sns.heatmap(
        data=df,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5,
    )

    plt.title(f"Predicted Score Distribution\n{home_team} vs {away_team}")
    plt.tight_layout()
    plt.show()
