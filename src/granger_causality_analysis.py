import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from time_series_analysis import COLOR, SEASONS  # reuse palette/season labels

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data"
TS_FOLDER = PROJECT_FOLDER / "outputs" / "time_series"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "granger"

CURRY_ENTRY_SEASON = "2009-10"  # first season in the Granger sample (Curry's rookie year)
MAX_LAG = 2
ALPHA = 0.05


# Curry's 3PA per game for every season he has played
def load_curry_3pa():
    career = pd.read_csv(DATA_FOLDER / "players" / "stephen_curry_career.csv")
    career = career[career["SEASON_ID"].isin(SEASONS)]
    per_game = career["FG3A"] / career["GP"]
    return pd.Series(per_game.values, index=career["SEASON_ID"].values, name="CURRY_3PA_PER_GAME")


# League-wide 3PA per game for every season
def load_league_3pa():
    return pd.read_csv(TS_FOLDER / "league_3pa_per_season.csv", index_col=0)["LEAGUE_3PA_PER_GAME"]


# Augmented Dickey-Fuller test
def adf_test(x):
    p_value = adfuller(x, autolag="AIC")[1]
    return p_value < ALPHA, p_value


# Difference until ADF finds stationarity
def resolve_stationarity(x, name, max_diffs=2):
    diffs, p_values = 0, []
    stationary, p = adf_test(x)
    p_values.append(p)
    while not stationary and diffs < max_diffs and len(x) - diffs > 3:
        diffs += 1
        stationary, p = adf_test(np.diff(x, n=diffs))
        p_values.append(p)

    print(
        f"  {name}: {'stationary' if stationary else 'NON-STATIONARY'} after {diffs} differencing "
        f"(ADF p-value(s): {', '.join(f'{v:.4f}' for v in p_values)})"
    )
    return diffs, stationary


def style_axis(ax):
    ax.grid(axis="y", color=COLOR["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR["baseline"])
    ax.tick_params(colors=COLOR["muted"])


def plot_granger_pvalues(p_values_by_lag):
    lags = sorted(p_values_by_lag)
    p_values = [p_values_by_lag[lag] for lag in lags]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor(COLOR["surface"])
    ax.set_facecolor(COLOR["surface"])

    colors = [COLOR["highlight"] if p < ALPHA else COLOR["blue"] for p in p_values]
    ax.bar(lags, p_values, color=colors, width=0.5, zorder=3)
    ax.axhline(ALPHA, color=COLOR["secondary_ink"], linestyle="--", linewidth=1.3, zorder=4, label=f"alpha = {ALPHA}")

    ax.set_xticks(lags)
    ax.set_xlabel("Lag (seasons)", color=COLOR["primary_ink"])
    ax.set_ylabel("Granger causality p-value", color=COLOR["primary_ink"])
    ax.set_title(
        "Granger causality p-values by lag\n(Curry's 3PA/game predicting league-wide 3PA/game)",
        color=COLOR["primary_ink"], fontsize=12, fontweight="bold", pad=14,
    )
    style_axis(ax)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR["secondary_ink"])

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "granger_causality_pvalues.png", facecolor=fig.get_facecolor())
    plt.close(fig)


# Run ADF stationarity checks then Granger causality at lag 1 and 2
def analyze_granger_causality():
    curry = load_curry_3pa()
    league = load_league_3pa()

    # Restrict to seasons Curry has actually played, per the proposal's scope for this objective
    seasons = [s for s in SEASONS if s >= CURRY_ENTRY_SEASON and s in curry.index]
    curry = curry.reindex(seasons)
    league = league.reindex(seasons)
    print(f"Granger causality sample: {seasons[0]} to {seasons[-1]} ({len(seasons)} seasons)")

    print("\nAugmented Dickey-Fuller stationarity tests:")
    curry_diffs, curry_stationary = resolve_stationarity(curry.values, "Curry 3PA/game")
    league_diffs, league_stationary = resolve_stationarity(league.values, "League 3PA/game")

    # Both series must be differenced by the same amount
    n_diffs = max(curry_diffs, league_diffs)
    curry_final = np.diff(curry.values, n=n_diffs) if n_diffs else curry.values
    league_final = np.diff(league.values, n=n_diffs) if n_diffs else league.values
    stationarity_resolved = curry_stationary and league_stationary

    data = np.column_stack([league_final, curry_final])
    print(
        f"\nGranger causality test (Curry's 3PA/game -> league-wide 3PA/game), "
        f"{n_diffs}x differenced, max lag {MAX_LAG}:"
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        test_results = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)

    rows = []
    p_values_by_lag = {}
    for lag in range(1, MAX_LAG + 1):
        f_stat, p_value = test_results[lag][0]["ssr_ftest"][:2]
        p_values_by_lag[lag] = p_value
        rows.append({
            "LAG": lag,
            "F_STATISTIC": f_stat,
            "P_VALUE": p_value,
            "SIGNIFICANT_AT_0_05": p_value < 0.05,
            "SIGNIFICANT_AT_0_10": p_value < 0.10,
            "STATIONARITY_RESOLVED": stationarity_resolved,
            "N_DIFFERENCES_APPLIED": n_diffs,
        })
        print(
            f"  Lag {lag}: F={f_stat:.3f}, p={p_value:.4f} "
            f"({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05)"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_FOLDER / "granger_causality_summary.csv", index=False)
    plot_granger_pvalues(p_values_by_lag)

    strong = any(row["P_VALUE"] < 0.05 for row in rows)
    weak = not strong and any(row["P_VALUE"] < 0.10 for row in rows)
    verdict = "STRONG (p<0.05 at lag 1 or 2)" if strong else "WEAK (p<0.10 at some lag)" if weak else "FAIL (p>0.10 at all lags)"
    print(f"\nOverall result: {verdict}")
    if not stationarity_resolved:
        print("WARNING: stationarity was not fully resolved after differencing; treat results with caution.")


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    analyze_granger_causality()
    print("Saved outputs to", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
