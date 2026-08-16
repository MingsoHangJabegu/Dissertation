import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from time_series_analysis import COLOR, SEASONS  # reuse palette/season labels

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
TS_FOLDER = PROJECT_FOLDER / "outputs" / "time_series"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "changepoint"

MVP_SEASON = "2015-16"
MIN_SEGMENT_SIZE = 2  # minimum seasons per segment
BIC_PENALTY_MULTIPLIER = 2.0  # multiplies the log(n) BIC penalty
N_RANDOM_INTERVALS = 200  # number of random intervals drawn for wild binary segmentation
RANDOM_INTERVAL_SEED = 0


# Residual sum of squares of a segment around its own mean
def segment_rss(x):
    if len(x) < 2:
        return 0.0
    return float(np.sum((x - x.mean()) ** 2))


# Search every valid split of `x` and return the index (and combined RSS) that minimises RSS across both halves
def best_split(x, min_size):
    n = len(x)
    best_i, best_rss = None, np.inf
    for i in range(min_size, n - min_size + 1):
        rss = segment_rss(x[:i]) + segment_rss(x[i:])
        if rss < best_rss:
            best_rss, best_i = rss, i
    return best_i, best_rss


# Select the split point for active segment [start, end) 
def binseg_select_split(x, start, end, min_size, all_intervals=None):
    split_i, _ = best_split(x[start:end], min_size)
    return None if split_i is None else start + split_i


# Draw M random intervals
def draw_random_intervals(n, min_size, n_intervals=N_RANDOM_INTERVALS, seed=RANDOM_INTERVAL_SEED):
    rng = np.random.default_rng(seed)
    span = 2 * min_size
    if n < span:
        return []

    intervals = []
    attempts = 0
    while len(intervals) < n_intervals and attempts < n_intervals * 20:
        attempts += 1
        s, e = sorted(rng.integers(0, n + 1, size=2))
        if e - s >= span:
            intervals.append((int(s), int(e)))
    return intervals


# Select the split point for active segment [start, end) using wild binary segmentation
def wbs_select_split(x, start, end, min_size, all_intervals):
    contained = [(s, e) for s, e in all_intervals if s >= start and e <= end]
    if not contained:
        contained = [(start, end)]

    best_b, best_local_gain = None, -np.inf
    for s, e in contained:
        seg = x[s:e]
        split_i, split_rss = best_split(seg, min_size)
        if split_i is None:
            continue
        b = s + split_i
        if b - start < min_size or end - b < min_size:
            continue
        local_gain = segment_rss(seg) - split_rss
        if local_gain > best_local_gain:
            best_local_gain, best_b = local_gain, b

    return best_b


# Repeatedly split whichever active segment yields the largest true RSS reduction
def _run_segmentation(x, min_size, penalty_multiplier, select_split, all_intervals=None):
    n = len(x)
    sigma2 = np.var(x)
    penalty = sigma2 * np.log(n) * penalty_multiplier

    segments = [(0, n)]
    breakpoints = []

    while True:
        best = None  # (gain, start, end, split_index)
        for start, end in segments:
            if end - start < 2 * min_size:
                continue
            split = select_split(x, start, end, min_size, all_intervals)
            if split is None:
                continue
            gain = segment_rss(x[start:end]) - (segment_rss(x[start:split]) + segment_rss(x[split:end]))
            if best is None or gain > best[0]:
                best = (gain, start, end, split)

        if best is None or best[0] < penalty:
            break

        gain, start, end, split = best
        breakpoints.append(split)
        segments.remove((start, end))
        segments.append((start, split))
        segments.append((split, end))

    return sorted(breakpoints)


# Binary segmentation
def binary_segmentation(x, min_size=MIN_SEGMENT_SIZE, penalty_multiplier=BIC_PENALTY_MULTIPLIER):
    x = np.asarray(x, dtype=float)
    return _run_segmentation(x, min_size, penalty_multiplier, binseg_select_split)


# Wild binary segmentation
def wild_binary_segmentation(
    x, min_size=MIN_SEGMENT_SIZE, penalty_multiplier=BIC_PENALTY_MULTIPLIER,
    n_intervals=N_RANDOM_INTERVALS, interval_seed=RANDOM_INTERVAL_SEED,
):
    x = np.asarray(x, dtype=float)
    all_intervals = draw_random_intervals(len(x), min_size, n_intervals, interval_seed)
    return _run_segmentation(x, min_size, penalty_multiplier, wbs_select_split, all_intervals)


# Mean of each segment implied by a sorted list of breakpoints
def segment_means(x, breakpoints):
    edges = [0] + list(breakpoints) + [len(x)]
    return [float(np.mean(x[edges[i]:edges[i + 1]])) for i in range(len(edges) - 1)]


# RSS reduction from the single best split of x (BinSeg test statistic)
def max_gain_statistic(x, min_size=MIN_SEGMENT_SIZE):
    split_i, split_rss = best_split(x, min_size)
    if split_i is None:
        return 0.0
    return segment_rss(x) - split_rss


# Permutation test for BinSeg
def permutation_test(x, min_size=MIN_SEGMENT_SIZE, n_permutations=20000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    observed = max_gain_statistic(x, min_size)

    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(x)
        if max_gain_statistic(shuffled, min_size) >= observed:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


# WBS test statistic
def wbs_max_gain_statistic(x, min_size, all_intervals):
    n = len(x)
    split = wbs_select_split(x, 0, n, min_size, all_intervals)
    if split is None:
        return 0.0
    return segment_rss(x) - (segment_rss(x[:split]) + segment_rss(x[split:]))


# Permutation test for wild binary segmentation
def wbs_permutation_test(
    x, min_size=MIN_SEGMENT_SIZE, n_permutations=20000, seed=0,
    n_intervals=N_RANDOM_INTERVALS, interval_seed=RANDOM_INTERVAL_SEED,
):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    all_intervals = draw_random_intervals(len(x), min_size, n_intervals, interval_seed)
    observed = wbs_max_gain_statistic(x, min_size, all_intervals)

    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(x)
        if wbs_max_gain_statistic(shuffled, min_size, all_intervals) >= observed:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


def style_axis(ax):
    ax.grid(axis="y", color=COLOR["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR["baseline"])
    ax.tick_params(colors=COLOR["muted"])


def plot_changepoints(league, breakpoints, means, method_label, filename):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(COLOR["surface"])
    ax.set_facecolor(COLOR["surface"])

    x = range(len(league))
    ax.plot(x, league.values, color=COLOR["blue"], linewidth=2, marker="o", markersize=4, label="League-wide 3PA/game")

    edges = [0] + list(breakpoints) + [len(league)]
    for i in range(len(edges) - 1):
        seg_start, seg_end = edges[i], edges[i + 1]
        ax.hlines(
            means[i], seg_start - 0.5, seg_end - 0.5,
            color=COLOR["highlight"], linewidth=2.5, alpha=0.9, zorder=3,
            label="Segment mean" if i == 0 else None,
        )

    for bp in breakpoints:
        ax.axvline(
            bp - 0.5, color=COLOR["secondary_ink"], linestyle="--", linewidth=1.3, alpha=0.8,
            label=f"Breakpoint: {SEASONS[bp - 1]} → {SEASONS[bp]}",
        )

    mvp_x = SEASONS.index(MVP_SEASON)
    ax.axvline(mvp_x, color=COLOR["muted"], linestyle=":", linewidth=1.5, alpha=0.9, label=f"Curry MVP season ({MVP_SEASON})")

    ax.set_xticks(list(x))
    ax.set_xticklabels(SEASONS, rotation=45, ha="right", fontsize=8, color=COLOR["secondary_ink"])
    ax.set_yticks(range(15, 41, 5))
    ax.set_ylim(15, 40)
    ax.set_ylabel("League-wide 3PA per game", color=COLOR["primary_ink"])
    ax.set_title(
        f"{method_label} change-point detection – league-wide 3PA per game",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold", pad=14,
    )
    style_axis(ax)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR["secondary_ink"])

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / filename, facecolor=fig.get_facecolor())
    plt.close(fig)


# Plotting function for the team-by-team change-point detection
def plot_team_changepoints(team_pivot, results, method_label, filename):
    teams_sorted = sorted(team_pivot.columns)
    n_teams = len(teams_sorted)
    ncols = 5
    nrows = -(-n_teams // ncols)  # ceil division

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), dpi=150,
        sharex=True, sharey=True,
    )
    fig.patch.set_facecolor(COLOR["surface"])

    x = range(len(SEASONS))
    mvp_x = SEASONS.index(MVP_SEASON)
    tick_idx = list(range(0, len(SEASONS), 4))

    # Bottommost populated axis in each column
    bottom_indices = {i % ncols: i for i in range(n_teams)}.values()

    for i, (ax, team) in enumerate(zip(axes.flat, teams_sorted)):
        ax.set_facecolor(COLOR["surface"])
        series = team_pivot[team].values
        breakpoints, means = results[team]

        ax.plot(x, series, color=COLOR["blue"], linewidth=1.2)

        edges = [0] + list(breakpoints) + [len(series)]
        for j in range(len(edges) - 1):
            seg_start, seg_end = edges[j], edges[j + 1]
            ax.hlines(
                means[j], seg_start - 0.5, seg_end - 0.5,
                color=COLOR["highlight"], linewidth=1.8, alpha=0.9, zorder=3,
            )

        for bp in breakpoints:
            ax.axvline(bp - 0.5, color=COLOR["secondary_ink"], linestyle="--", linewidth=1, alpha=0.8)
        ax.axvline(mvp_x, color=COLOR["muted"], linestyle=":", linewidth=1, alpha=0.8)

        ax.set_title(team, fontsize=9, fontweight="bold", color=COLOR["primary_ink"])
        ax.grid(axis="y", color=COLOR["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6, colors=COLOR["muted"])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR["baseline"])

        ax.set_xticks(tick_idx)
        if i in bottom_indices:
            ax.set_xticklabels([SEASONS[j] for j in tick_idx], rotation=45, ha="right", fontsize=6)
        else:
            ax.set_xticklabels([])

    for ax in axes.flat[n_teams:]:
        ax.set_visible(False)

    fig.supxlabel("Season", color=COLOR["secondary_ink"], fontsize=10)
    fig.supylabel("3PA per game", color=COLOR["secondary_ink"], fontsize=9)
    fig.suptitle(
        f"Team-by-team {method_label.lower()} change-point detection, 2004-05 to 2024-25\n"
        f"(dotted line marks Curry's {MVP_SEASON} MVP season; dashed lines mark detected breakpoints)",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0.015, 0.015, 1, 0.92))
    fig.savefig(OUTPUT_FOLDER / filename, facecolor=fig.get_facecolor())
    plt.close(fig)


# Plotting function for the tally of team breakpoints by season transition
def plot_team_breakpoint_histogram(summary, method_label, filename):
    counts = summary["BREAKPOINT_INDEX"].value_counts().reindex(range(1, len(SEASONS)), fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    fig.patch.set_facecolor(COLOR["surface"])
    ax.set_facecolor(COLOR["surface"])

    ax.bar(counts.index, counts.values, color=COLOR["blue"], width=0.7, zorder=3)

    mvp_x = SEASONS.index(MVP_SEASON)
    ax.axvline(
        mvp_x, color=COLOR["muted"], linestyle=":", linewidth=1.5, alpha=0.9,
        label=f"Curry MVP season ({MVP_SEASON})",
    )

    ax.set_xticks(range(1, len(SEASONS)))
    ax.set_xticklabels(SEASONS[1:], rotation=45, ha="right", fontsize=8, color=COLOR["secondary_ink"])
    ax.set_yticks(range(0, int(counts.max()) + 2))
    ax.set_ylabel("Number of teams", color=COLOR["primary_ink"])
    ax.set_xlabel("Season a breakpoint was detected in (i.e. first season of the new segment)", color=COLOR["primary_ink"])
    ax.set_title(
        f"Distribution of team-level breakpoints by season ({method_label})",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold", pad=14,
    )
    style_axis(ax)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR["secondary_ink"])

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / filename, facecolor=fig.get_facecolor())
    plt.close(fig)


# Segmentation methods and their matching tests
METHODS = {
    "binseg": {
        "label": "Binary segmentation",
        "suffix": "binseg",
        "segment": binary_segmentation,
        "test": permutation_test,
    },
    "wbs": {
        "label": "Wild binary segmentation",
        "suffix": "wbs",
        "segment": wild_binary_segmentation,
        "test": wbs_permutation_test,
    },
}


# Run and save
def analyze_team_changepoints(method):
    team_pivot = pd.read_csv(TS_FOLDER / "team_3pa_per_season.csv", index_col=0)
    team_pivot = team_pivot.reindex(SEASONS)
    mvp_index = SEASONS.index(MVP_SEASON)

    rows = []
    results = {}
    n_significant = 0
    for team in team_pivot.columns:
        series = team_pivot[team].values
        breakpoints = method["segment"](series)
        means = segment_means(series, breakpoints)
        results[team] = (breakpoints, means)

        # One permutation test per team
        _, p_value = method["test"](series, n_permutations=20000, seed=0)
        significant = p_value < 0.05
        n_significant += significant

        for i, bp in enumerate(breakpoints):
            distance_from_mvp = abs(bp - mvp_index)
            rows.append({
                "TEAM": team,
                "BREAKPOINT_INDEX": bp,
                "BETWEEN_SEASONS": f"{SEASONS[bp - 1]} -> {SEASONS[bp]}",
                "SEGMENT_MEAN_BEFORE": means[i],
                "SEGMENT_MEAN_AFTER": means[i + 1],
                "MEAN_CHANGE": means[i + 1] - means[i],
                "SEASONS_FROM_CURRY_MVP_2015_16": distance_from_mvp,
                "WITHIN_TWO_SEASONS_OF_MVP": distance_from_mvp <= 2,
                "PERMUTATION_P_VALUE": p_value,
                "SIGNIFICANT_AT_0_05": significant,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_FOLDER / f"team_3pa_changepoints_summary_{method['suffix']}.csv", index=False)
    plot_team_changepoints(team_pivot, results, method["label"], f"team_3pa_changepoints_{method['suffix']}.png")
    plot_team_breakpoint_histogram(summary, method["label"], f"team_breakpoint_histogram_{method['suffix']}.png")

    within = int(summary["WITHIN_TWO_SEASONS_OF_MVP"].sum()) if not summary.empty else 0
    print(
        f"[{method['label']}] Team change-point detection: {len(rows)} breakpoint(s) across "
        f"{len(team_pivot.columns)} teams ({within} within two seasons of Curry's {MVP_SEASON} MVP season)"
    )
    print(
        f"  Permutation test (per team, 5000 permutations): "
        f"{n_significant}/{len(team_pivot.columns)} teams significant at alpha=0.05"
    )


# Run and save
def analyze_league_changepoints(method):
    league = pd.read_csv(TS_FOLDER / "league_3pa_per_season.csv", index_col=0)["LEAGUE_3PA_PER_GAME"]
    league.index = range(len(league))

    breakpoints = method["segment"](league.values)
    means = segment_means(league.values, breakpoints)
    mvp_index = SEASONS.index(MVP_SEASON)

    rows = []
    for i, bp in enumerate(breakpoints):
        distance_from_mvp = abs(bp - mvp_index)
        rows.append({
            "BREAKPOINT_INDEX": bp,
            "BETWEEN_SEASONS": f"{SEASONS[bp - 1]} -> {SEASONS[bp]}",
            "SEGMENT_MEAN_BEFORE": means[i],
            "SEGMENT_MEAN_AFTER": means[i + 1],
            "MEAN_CHANGE": means[i + 1] - means[i],
            "SEASONS_FROM_CURRY_MVP_2015_16": distance_from_mvp,
            "WITHIN_TWO_SEASONS_OF_MVP": distance_from_mvp <= 2,
        })

    observed_gain, p_value = method["test"](league.values)
    for row in rows:
        row["PERMUTATION_P_VALUE"] = p_value
        row["SIGNIFICANT_AT_0_05"] = p_value < 0.05

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_FOLDER / f"league_3pa_changepoints_summary_{method['suffix']}.csv", index=False)
    plot_changepoints(league, breakpoints, means, method["label"], f"league_3pa_changepoints_{method['suffix']}.png")

    print(f"[{method['label']}] Change-point detection: found {len(breakpoints)} breakpoint(s)")
    for row in rows:
        print(
            f"  {row['BETWEEN_SEASONS']} (mean {row['SEGMENT_MEAN_BEFORE']:.2f} -> {row['SEGMENT_MEAN_AFTER']:.2f}, "
            f"{row['SEASONS_FROM_CURRY_MVP_2015_16']} season(s) from Curry's {MVP_SEASON} MVP season)"
        )
    print(
        f"  Permutation test: observed RSS reduction = {observed_gain:.2f}, p-value = {p_value:.4f} "
        f"({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05, 10000 permutations)"
    )


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    for method in METHODS.values():
        analyze_league_changepoints(method)
        analyze_team_changepoints(method)
    print("Saved outputs to", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
