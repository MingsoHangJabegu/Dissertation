from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "time_series"

SEASONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2004, 2025)]

GSW_TEAM_ID = 1610612744

# Curry milestone seasons
MILESTONES = {
    "2015-16": "MVP season 2\n(unanimous, 402 3PM)",
    "2009-10": "League entry",
    "2012-13": "Breakout season\n(272 3PM, then-record)",
    "2014-15": "MVP season 1",
}

# NBA All-Star Game date for each season
ALLSTAR_GAME_DATES = {
    "2004-05": "2005-02-20",
    "2005-06": "2006-02-19",
    "2006-07": "2007-02-18",
    "2007-08": "2008-02-17",
    "2008-09": "2009-02-15",
    "2009-10": "2010-02-14",
    "2010-11": "2011-02-20",
    "2011-12": "2012-02-26",
    "2012-13": "2013-02-17",
    "2013-14": "2014-02-16",
    "2014-15": "2015-02-15",
    "2015-16": "2016-02-14",
    "2016-17": "2017-02-19",
    "2017-18": "2018-02-18",
    "2018-19": "2019-02-17",
    "2019-20": "2020-02-16",
    "2020-21": "2021-03-07",
    "2021-22": "2022-02-20",
    "2022-23": "2023-02-19",
    "2023-24": "2024-02-18",
    "2024-25": "2025-02-16",
}

# Color palette for charts
COLOR = {
    "surface": "#fcfcfb",
    "primary_ink": "#0b0b0b",
    "secondary_ink": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "blue": "#2a78d6",
    "context_gray": "#c3c2b7",
    "highlight": "#d6822a",
}

# Team highlighted for its jump
JUMP_TEAM = "Houston Rockets"

# Official primary colors for the teams highlighted
TEAM_COLORS = {
    "Golden State Warriors": "#1D428A",
    "Houston Rockets": "#CE1141",
    "Boston Celtics": "#007A33",
}


# Load every season's team stats
def load_team_stats():
    frames = []
    for season in SEASONS:
        df = pd.read_csv(DATA_FOLDER / "team_stats" / f"team_stats_{season}.csv")
        frames.append(df[["TEAM_ID", "TEAM_NAME", "GP", "FG3A", "SEASON"]])
    return pd.concat(frames, ignore_index=True)


# League-wide 3PA/game per season, weighted by each team's actual number of games played
def compute_league_trend(team_stats):

    # Each team's season total divided by league wide total games
    def weighted_mean(group):
        return (group["FG3A"] * group["GP"]).sum() / group["GP"].sum()

    league = team_stats.groupby("SEASON", sort=False).apply(
        weighted_mean, include_groups=False
    )
    return league.reindex(SEASONS)


# Team-by-team 3PA/game per season
def compute_team_trend(team_stats):
    # Pivot on TEAM_ID, then rename columns to the latest known team name since teams have changed names
    pivot = team_stats.pivot_table(index="SEASON", columns="TEAM_ID", values="FG3A")
    pivot = pivot.reindex(SEASONS)

    latest_names = (
        team_stats.sort_values("SEASON")
        .groupby("TEAM_ID")["TEAM_NAME"]
        .last()
    )
    pivot.columns = [latest_names[tid] for tid in pivot.columns]
    return pivot


# Load every season's individual team-game rows (exact per-game FG3A)
def load_game_logs():
    frames = []
    for season in SEASONS:
        df = pd.read_csv(
            DATA_FOLDER / "game_logs" / f"game_log_{season}.csv",
            usecols=["TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE", "FG3A", "SEASON"],
        )
        frames.append(df)
    game_logs = pd.concat(frames, ignore_index=True)
    game_logs["GAME_DATE"] = pd.to_datetime(game_logs["GAME_DATE"])
    return game_logs


# Team-by-team 3PA per game
def compute_team_matchweek_trend(game_logs):
    game_logs = game_logs.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    game_logs = game_logs.assign(
        GAME_NUM=game_logs.groupby(["TEAM_ID", "SEASON"]).cumcount() + 1
    )

    # Map each TEAM_ID to its latest known abbreviation
    latest_abbr = (
        game_logs.sort_values("SEASON").groupby("TEAM_ID")["TEAM_ABBREVIATION"].last()
    )
    game_logs = game_logs.assign(TEAM=game_logs["TEAM_ID"].map(latest_abbr))

    return game_logs.pivot_table(index="GAME_NUM", columns="TEAM", values="FG3A")


# Team-by-team 3PA per game expressed as a % deviation from that team's own season-average 3PA
def compute_team_matchweek_relative_trend(game_logs, team_stats):
    # Map each TEAM_ID/SEASON to that team's season average 3PA
    season_avg = team_stats.set_index(["TEAM_ID", "SEASON"])["FG3A"]

    # Compute each game as a % deviation from that team's own season average 3PA
    game_logs = game_logs.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    game_logs = game_logs.assign(
        GAME_NUM=game_logs.groupby(["TEAM_ID", "SEASON"]).cumcount() + 1
    )
    game_logs = game_logs.assign(
        SEASON_AVG=game_logs.set_index(["TEAM_ID", "SEASON"]).index.map(season_avg)
    )
    game_logs = game_logs.assign(
        REL_FG3A_PCT=(game_logs["FG3A"] - game_logs["SEASON_AVG"]) / game_logs["SEASON_AVG"] * 100
    )

    # Map each TEAM_ID to its latest known abbreviation
    latest_abbr = (
        game_logs.sort_values("SEASON").groupby("TEAM_ID")["TEAM_ABBREVIATION"].last()
    )
    game_logs = game_logs.assign(TEAM=game_logs["TEAM_ID"].map(latest_abbr))

    return game_logs.pivot_table(index="GAME_NUM", columns="TEAM", values="REL_FG3A_PCT")


# Find the typical game number of the All-Star break
def compute_allstar_break_game_numbers(game_logs):
    game_logs = game_logs.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    game_logs = game_logs.assign(
        GAME_NUM=game_logs.groupby(["TEAM_ID", "SEASON"]).cumcount() + 1,
        ALLSTAR_DATE=game_logs["SEASON"].map(ALLSTAR_GAME_DATES).pipe(pd.to_datetime),
    )

    # Only consider games before the All-Star break, then find the last game number for each team in each season
    before_break = game_logs[game_logs["GAME_DATE"] < game_logs["ALLSTAR_DATE"]]
    positions = before_break.groupby(["TEAM_ID", "SEASON"])["GAME_NUM"].max()
    return positions.median(), positions.quantile(0.25), positions.quantile(0.75)


# Annotate the league-wide 3PA/game trend chart with Curry milestone seasons
def annotate_milestones(ax, y_top):
    # Each label sits on its own horizontal band (y_top - i * band_height) to avoid overlapping text
    band_height = 3.2
    for i, (season, label) in enumerate(MILESTONES.items()):
        x = SEASONS.index(season)
        ax.axvline(x=x, color=COLOR["muted"], linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate(
            label,
            xy=(x, y_top - i * band_height),
            xytext=(x + 0.3, y_top - i * band_height),
            fontsize=8,
            color=COLOR["secondary_ink"],
            ha="left",
            va="top",
        )

# Plotting functions for the league-wide and team-by-team 3PA/game trends
def plot_league_trend(league):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(COLOR["surface"])
    ax.set_facecolor(COLOR["surface"])

    x = range(len(SEASONS))
    ax.plot(x, league.values, color=COLOR["blue"], linewidth=2.5, marker="o", markersize=4)

    y_top = league.max() + 3
    annotate_milestones(ax, y_top)

    ax.set_xticks(list(x))
    ax.set_xticklabels(SEASONS, rotation=45, ha="right", fontsize=8, color=COLOR["secondary_ink"])
    ax.set_ylabel("League-wide 3PA per game", color=COLOR["primary_ink"])
    ax.set_title(
        "NBA three-point attempts per game, 2004-05 to 2024-25",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold", pad=14,
    )
    ax.grid(axis="y", color=COLOR["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR["baseline"])
    ax.tick_params(colors=COLOR["muted"])
    ax.set_ylim(top=y_top + 1.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "league_3pa_trend.png", facecolor=fig.get_facecolor())
    plt.close(fig)


# Plotting function for the team-by-team 3PA/game trend chart, highlighting Golden State Warriors
def plot_team_trend(team_pivot, league):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(COLOR["surface"])
    ax.set_facecolor(COLOR["surface"])

    x = range(len(SEASONS))
    gsw_name = next(col for col in team_pivot.columns if "Golden State" in col)
    gsw_last_season_3pa = team_pivot[gsw_name].iloc[-1]

    # Teams whose most recent season 3PA/game exceeds GSW's, highlighted alongside GSW
    exceed_teams = [
        col for col in team_pivot.columns
        if col != gsw_name and team_pivot[col].iloc[-1] > gsw_last_season_3pa
    ]

    for col in team_pivot.columns:
        if col in (gsw_name, JUMP_TEAM) or col in exceed_teams:
            continue
        ax.plot(x, team_pivot[col].values, color=COLOR["context_gray"], linewidth=1, alpha=0.6)

    for col in exceed_teams:
        ax.plot(
            x, team_pivot[col].values,
            color=TEAM_COLORS.get(col, COLOR["highlight"]), linewidth=1.5, alpha=0.85, zorder=3,
            label=f"{col} (higher {SEASONS[-1]} 3PA/game than GSW)",
        )

    ax.plot(
        x, team_pivot[JUMP_TEAM].values,
        color=TEAM_COLORS.get(JUMP_TEAM, COLOR["highlight"]), linewidth=2, alpha=0.9, zorder=4,
        label=JUMP_TEAM,
    )

    ax.plot(
        x, team_pivot[gsw_name].values,
        color=TEAM_COLORS.get(gsw_name, COLOR["blue"]), linewidth=2.5, label="Golden State Warriors", zorder=5,
    )
    ax.plot(
        x, league.values,
        color=COLOR["secondary_ink"], linewidth=1.5, linestyle="--",
        label="League average", zorder=4,
    )
    # proxy line for the legend entry representing all other, non-highlighted teams
    other_count = len(team_pivot.columns) - 2 - len(exceed_teams)
    ax.plot([], [], color=COLOR["context_gray"], linewidth=1.5, label=f"Other NBA teams ({other_count})")

    ax.set_xticks(list(x))
    ax.set_xticklabels(SEASONS, rotation=45, ha="right", fontsize=8, color=COLOR["secondary_ink"])
    ax.set_ylabel("3PA per game", color=COLOR["primary_ink"])
    ax.set_title(
        "Team-by-team three-point attempts per game, 2004-05 to 2024-25",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold", pad=14,
    )
    ax.grid(axis="y", color=COLOR["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR["baseline"])
    ax.tick_params(colors=COLOR["muted"])
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR["secondary_ink"])

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "team_3pa_trend.png", facecolor=fig.get_facecolor())
    plt.close(fig)


# Plotting function for the team-by-team 3PA/game by point in the season
def plot_team_matchweek_trend(matchweek_pivot, allstar_median, allstar_q1, allstar_q3):
    teams_sorted = sorted(matchweek_pivot.columns)
    ncols = 5
    nrows = -(-len(teams_sorted) // ncols)  # ceil division

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), dpi=150,
        sharex=True, sharey=True,
    )
    fig.patch.set_facecolor(COLOR["surface"])

    for ax, team in zip(axes.flat, teams_sorted):
        ax.set_facecolor(COLOR["surface"])
        series = matchweek_pivot[team].dropna()
        ax.axvspan(allstar_q1, allstar_q3, color=COLOR["muted"], alpha=0.15, linewidth=0)
        ax.axvline(allstar_median, color=COLOR["muted"], linestyle="--", linewidth=1, alpha=0.8)
        ax.plot(series.index, series.values, color=COLOR["blue"], linewidth=1.2)
        ax.set_title(team, fontsize=9, fontweight="bold", color=COLOR["primary_ink"])
        ax.grid(color=COLOR["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6, colors=COLOR["muted"])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR["baseline"])

    for ax in axes.flat[len(teams_sorted):]:
        ax.set_visible(False)

    fig.supxlabel("Game number in season", color=COLOR["secondary_ink"], fontsize=10)
    fig.supylabel("3PA per game\n(avg across seasons at this game number)", color=COLOR["secondary_ink"], fontsize=9)
    fig.suptitle(
        "Team 3PA per game by point in the season, 2004-05 to 2024-25\n"
        "(dashed line/band marks the typical All-Star break)",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0.015, 0.015, 1, 0.92))
    fig.savefig(OUTPUT_FOLDER / "team_matchweek_trend.png", facecolor=fig.get_facecolor())
    plt.close(fig)


# Plotting function for the team-by-team 3PA/game relative to own season average by point in the season
def plot_team_matchweek_relative_trend(matchweek_pivot, allstar_median, allstar_q1, allstar_q3):
    teams_sorted = sorted(matchweek_pivot.columns)
    ncols = 5
    nrows = -(-len(teams_sorted) // ncols)  # ceil division

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), dpi=150,
        sharex=True, sharey=True,
    )
    fig.patch.set_facecolor(COLOR["surface"])

    smooth_window = 5
    for ax, team in zip(axes.flat, teams_sorted):
        ax.set_facecolor(COLOR["surface"])
        series = matchweek_pivot[team].dropna()
        smoothed = series.rolling(smooth_window, center=True, min_periods=1).mean()
        ax.axvspan(allstar_q1, allstar_q3, color=COLOR["muted"], alpha=0.15, linewidth=0)
        ax.axvline(allstar_median, color=COLOR["muted"], linestyle="--", linewidth=1, alpha=0.8)
        ax.axhline(0, color=COLOR["baseline"], linewidth=1)
        ax.plot(series.index, series.values, color=COLOR["blue"], linewidth=0.8, alpha=0.25)
        ax.plot(smoothed.index, smoothed.values, color=COLOR["blue"], linewidth=1.6)
        ax.set_title(team, fontsize=9, fontweight="bold", color=COLOR["primary_ink"])
        ax.grid(color=COLOR["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6, colors=COLOR["muted"])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR["baseline"])

    for ax in axes.flat[len(teams_sorted):]:
        ax.set_visible(False)

    fig.supxlabel("Game number in season", color=COLOR["secondary_ink"], fontsize=10)
    fig.supylabel("3PA vs own season average (%)", color=COLOR["secondary_ink"], fontsize=9)
    fig.suptitle(
        "Team 3PA per game relative to own season average, by point in the season, 2004-05 to 2024-25\n"
        f"(bold line: {smooth_window}-game rolling average; faint line: raw per-game value; "
        "dashed line/band marks the typical All-Star break)",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0.015, 0.015, 1, 0.92))
    fig.savefig(OUTPUT_FOLDER / "team_matchweek_relative_trend.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    team_stats = load_team_stats()

    league = compute_league_trend(team_stats)
    team_pivot = compute_team_trend(team_stats)

    league.rename("LEAGUE_3PA_PER_GAME").to_csv(OUTPUT_FOLDER / "league_3pa_per_season.csv")
    team_pivot.to_csv(OUTPUT_FOLDER / "team_3pa_per_season.csv")

    plot_league_trend(league)
    plot_team_trend(team_pivot, league)

    game_logs = load_game_logs()
    matchweek_pivot = compute_team_matchweek_trend(game_logs)
    allstar_median, allstar_q1, allstar_q3 = compute_allstar_break_game_numbers(game_logs)

    matchweek_pivot.to_csv(OUTPUT_FOLDER / "team_3pa_per_game_number.csv")
    plot_team_matchweek_trend(matchweek_pivot, allstar_median, allstar_q1, allstar_q3)

    relative_matchweek_pivot = compute_team_matchweek_relative_trend(game_logs, team_stats)
    relative_matchweek_pivot.to_csv(OUTPUT_FOLDER / "team_3pa_relative_per_game_number.csv")
    plot_team_matchweek_relative_trend(relative_matchweek_pivot, allstar_median, allstar_q1, allstar_q3)

    print("Saved outputs to", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
