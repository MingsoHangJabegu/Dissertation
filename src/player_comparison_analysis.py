import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from time_series_analysis import COLOR, SEASONS  # reuse palette/season labels
from changepoint_analysis import (
    binary_segmentation, wild_binary_segmentation, segment_means, permutation_test, wbs_permutation_test,
)
from granger_causality_analysis import resolve_stationarity, ALPHA, MAX_LAG

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data"
TS_FOLDER = PROJECT_FOLDER / "outputs" / "time_series"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "player_comparison"

PENALTY_MULTIPLIER = 2.0

# Curry plus the four comparison players 
PLAYERS = ["Stephen Curry", "Klay Thompson", "James Harden", "Damian Lillard", "Kevin Durant"]

PLAYER_COLORS = {
    "Stephen Curry": COLOR["highlight"],
    "Klay Thompson": "#1D428A",   
    "James Harden": "#CE1141",    
    "Damian Lillard": "#00471B",  
    "Kevin Durant": "#1D1160",    
}

def player_filename(name):
    return name.lower().replace(" ", "_")


# Keep only the "TOT" row for players traded mid-season
def load_player_season_totals(name):
    career = pd.read_csv(DATA_FOLDER / "players" / f"{player_filename(name)}_career.csv")
    has_tot = career.groupby("SEASON_ID")["TEAM_ABBREVIATION"].transform(lambda s: (s == "TOT").any())
    career = career[~has_tot | (career["TEAM_ABBREVIATION"] == "TOT")]
    return career[career["SEASON_ID"].isin(SEASONS)]


# Season-by-season 3PA/game for one player (NaN for seasons not played)
def load_player_3pa(name):
    career = load_player_season_totals(name)
    per_game = pd.Series(
        (career["FG3A"] / career["GP"]).values, index=career["SEASON_ID"].values,
    )
    return per_game.reindex(SEASONS)


def load_league_3pa():
    return pd.read_csv(TS_FOLDER / "league_3pa_per_season.csv", index_col=0)["LEAGUE_3PA_PER_GAME"]


# Each player's peak season relative to the league
def compute_peak_relative_seasons(player_series, league_series):
    league = league_series.reindex(SEASONS)
    peaks = {}
    for name, series in player_series.items():
        ratio = (series.reindex(SEASONS) / league).dropna()
        if ratio.empty:
            continue
        season = ratio.idxmax()
        peaks[name] = (season, float(ratio.loc[season]))
    return peaks


# Segmentation methods and their matching permutation tests
CHANGEPOINT_METHODS = {
    "binseg": {
        "label": "Binary segmentation",
        "suffix": "binseg",
        "segment": binary_segmentation,
        "test": permutation_test
        },
    "wbs": {
        "label": "Wild binary segmentation",
        "suffix": "wbs",
        "segment": wild_binary_segmentation,
        "test": wbs_permutation_test
        },
}


# League-wide change-point detection at 2.0 penalty multiplier
def analyze_league_changepoints(method):
    league = load_league_3pa().reindex(SEASONS)
    values = league.values
    breakpoints = method["segment"](values, penalty_multiplier=PENALTY_MULTIPLIER)
    means = segment_means(values, breakpoints)
    _, p_value = method["test"](values, n_permutations=20000, seed=0)

    rows = []
    for i, bp in enumerate(breakpoints):
        rows.append({
            "METHOD": method["label"],
            "PENALTY_MULTIPLIER": PENALTY_MULTIPLIER,
            "BETWEEN_SEASONS": f"{SEASONS[bp - 1]} -> {SEASONS[bp]}",
            "SEGMENT_MEAN_BEFORE": means[i],
            "SEGMENT_MEAN_AFTER": means[i + 1],
            "MEAN_CHANGE": means[i + 1] - means[i],
            "PERMUTATION_P_VALUE": p_value,
            "SIGNIFICANT_AT_0_05": p_value < 0.05,
        })
    return rows, values, breakpoints, means, p_value


# ADF stationarity checks (differencing if needed)
# Granger causality
def analyze_player_granger(name, player_series, league_series):
    played = player_series.dropna()
    entry_season = played.index[0]
    seasons = [s for s in SEASONS if s >= entry_season]

    player_vals = player_series.reindex(seasons)
    league_vals = league_series.reindex(seasons)
    # Injury seasons with no games dropped from both series together so the two inputs stay aligned
    valid = player_vals.notna() & league_vals.notna()
    player_vals = player_vals[valid]
    league_vals = league_vals[valid]

    player_diffs, player_stationary = resolve_stationarity(player_vals.values, f"{name} 3PA/game")
    league_diffs, league_stationary = resolve_stationarity(league_vals.values, "League 3PA/game")
    n_diffs = max(player_diffs, league_diffs)
    player_final = np.diff(player_vals.values, n=n_diffs) if n_diffs else player_vals.values
    league_final = np.diff(league_vals.values, n=n_diffs) if n_diffs else league_vals.values
    stationarity_resolved = player_stationary and league_stationary

    data = np.column_stack([league_final, player_final])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        test_results = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)

    rows = []
    for lag in range(1, MAX_LAG + 1):
        f_stat, p_value = test_results[lag][0]["ssr_ftest"][:2]
        rows.append({
            "PLAYER": name,
            "LAG": lag,
            "F_STATISTIC": f_stat,
            "P_VALUE": p_value,
            "SIGNIFICANT_AT_0_05": p_value < 0.05,
            "SIGNIFICANT_AT_0_10": p_value < 0.10,
            "STATIONARITY_RESOLVED": stationarity_resolved,
            "N_DIFFERENCES_APPLIED": n_diffs,
            "N_SEASONS": len(player_vals),
        })
    return rows


def style_axis(ax):
    ax.grid(axis="y", color=COLOR["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR["baseline"])
    ax.tick_params(colors=COLOR["muted"])


# Single league-wide change-point plot for one segmentation method
def plot_league_changepoints(values, breakpoints, means, method_label, suffix, peak_seasons):
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(COLOR["surface"])
    ax.set_facecolor(COLOR["surface"])

    x = range(len(values))
    ax.plot(x, values, color=COLOR["blue"], linewidth=2, marker="o", markersize=4, label="League-wide 3PA/game", zorder=3)

    edges = [0] + list(breakpoints) + [len(values)]
    for i in range(len(edges) - 1):
        seg_start, seg_end = edges[i], edges[i + 1]
        ax.hlines(
            means[i], seg_start - 0.5, seg_end - 0.5,
            color=COLOR["highlight"], linewidth=2.5, alpha=0.9, zorder=4,
            label="Segment mean" if i == 0 else None,
        )
    for bp in breakpoints:
        ax.axvline(
            bp - 0.5, color=COLOR["muted"], linestyle="--", linewidth=1.3, alpha=0.8,
            label=f"Breakpoint: {SEASONS[bp - 1]} -> {SEASONS[bp]}",
        )

    for name, (season, ratio) in peak_seasons.items():
        season_x = SEASONS.index(season)
        style = "-." if name == "Stephen Curry" else ":"
        ax.axvline(
            season_x, color=PLAYER_COLORS[name], linestyle=style, linewidth=1.8, alpha=0.95, zorder=5,
            label=f"{name} peak vs league ({season})",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(SEASONS, rotation=45, ha="right", fontsize=8, color=COLOR["secondary_ink"])
    ax.set_ylabel("League-wide 3PA per game", color=COLOR["primary_ink"])
    ax.set_title(
        f"{method_label} change-point detection (penalty multiplier {PENALTY_MULTIPLIER}) - league-wide 3PA per game\n"
        "with each player's peak season relative to the league",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold", pad=14,
    )
    style_axis(ax)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(COLOR["secondary_ink"])

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / f"league_changepoints_{suffix}_pen2p0.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_players_granger_pvalues(granger_summary):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor(COLOR["surface"])
    ax.set_facecolor(COLOR["surface"])

    lags = list(range(1, MAX_LAG + 1))
    x = np.arange(len(lags))
    width = 0.8 / len(PLAYERS)

    for i, name in enumerate(PLAYERS):
        p_values = [
            granger_summary.query("PLAYER == @name and LAG == @lag")["P_VALUE"].iloc[0]
            for lag in lags
        ]
        offsets = x + (i - (len(PLAYERS) - 1) / 2) * width
        bars = ax.bar(offsets, p_values, width=width * 0.9, color=PLAYER_COLORS[name], zorder=3, label=name)
        for bar, p in zip(bars, p_values):
            if p < ALPHA:
                ax.annotate(
                    "*", (bar.get_x() + bar.get_width() / 2, p), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=12,
                    color=COLOR["primary_ink"], fontweight="bold",
                )

    ax.axhline(ALPHA, color=COLOR["secondary_ink"], linestyle="--", linewidth=1.3, zorder=4, label=f"alpha = {ALPHA}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Lag {lag}" for lag in lags])
    ax.set_xlabel("Lag (seasons)", color=COLOR["primary_ink"])
    ax.set_ylabel("Granger causality p-value", color=COLOR["primary_ink"])
    ax.set_title(
        "Granger causality p-values by lag and player\n(* = significant at alpha=0.05)",
        color=COLOR["primary_ink"], fontsize=12, fontweight="bold", pad=14,
    )
    style_axis(ax)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    for text in legend.get_texts():
        text.set_color(COLOR["secondary_ink"])

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "players_granger_pvalues.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def analyze_players():
    league = load_league_3pa()
    player_series = {name: load_player_3pa(name) for name in PLAYERS}

    peak_seasons = compute_peak_relative_seasons(player_series, league)
    print("Peak season relative to the league (player 3PA/game / league-average team 3PA/game that season):")
    for name, (season, ratio) in peak_seasons.items():
        print(f"  {name}: {season} ({ratio:.2f}x league-average team)")

    print(f"\nLeague-wide change-point detection at penalty multiplier {PENALTY_MULTIPLIER}:")
    changepoint_rows = []
    for method in CHANGEPOINT_METHODS.values():
        rows, values, breakpoints, means, p_value = analyze_league_changepoints(method)
        changepoint_rows += rows
        plot_league_changepoints(values, breakpoints, means, method["label"], method["suffix"], peak_seasons)
        print(
            f"  [{method['label']}] {len(breakpoints)} breakpoint(s), "
            f"permutation p-value = {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05)"
        )
    pd.DataFrame(changepoint_rows).to_csv(OUTPUT_FOLDER / "league_changepoints_summary.csv", index=False)

    print("\nGranger causality per player (season-level 3PA/game -> league-wide 3PA/game):")
    granger_rows = []
    for name in PLAYERS:
        rows = analyze_player_granger(name, player_series[name], league)
        granger_rows += rows
        for row in rows:
            print(
                f"  {name} lag {row['LAG']}: F={row['F_STATISTIC']:.3f}, p={row['P_VALUE']:.4f} "
                f"({'significant' if row['SIGNIFICANT_AT_0_05'] else 'not significant'} at alpha=0.05)"
            )
    granger_summary = pd.DataFrame(granger_rows)
    granger_summary.to_csv(OUTPUT_FOLDER / "players_granger_summary.csv", index=False)
    plot_players_granger_pvalues(granger_summary)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    analyze_players()
    print("\nSaved outputs to", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
