import itertools
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent))
from time_series_analysis import COLOR, SEASONS, TEAM_COLORS

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
TS_FOLDER = PROJECT_FOLDER / "outputs" / "time_series"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "arima"

FORECAST_SEASONS = 3
P_RANGE, Q_RANGE = range(4), range(4)
MAX_D = 1
ADF_ALPHA = 0.05

ABBR_TO_NAME = {
    "ATL": "Atlanta Hawks",
    "BKN": "Brooklyn Nets",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}
NAME_TO_ABBR = {name: abbr for abbr, name in ABBR_TO_NAME.items()}

warnings.filterwarnings("ignore")


# Pick d via the Augmented Dickey-Fuller test
def select_d(series, max_d=MAX_D, alpha=ADF_ALPHA):
    diffed = series.dropna()
    for d in range(max_d + 1):
        try:
            pvalue = adfuller(diffed, autolag="AIC")[1]
        except Exception as exc:
            print(f"select_d: ADF test failed at d={d} ({exc}); defaulting to d={max_d}")
            return max_d
        if pvalue < alpha:
            return d
        diffed = diffed.diff().dropna()
    return max_d


# Grid-search (p, q) by AICc with d fixed by select_d; seasonal order fixed at (0, 0, 0, 0)
def fit_best_order(series):
    d = select_d(series)
    best = None
    attempts = []
    for p, q in itertools.product(P_RANGE, Q_RANGE):
        if p == 0 and q == 0:
            continue
        try:
            fit = SARIMAX(
                series, order=(p, d, q), trend="c",
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
        except Exception:
            attempts.append({"ORDER": (p, d, q), "AICC": float("nan")})
            continue
        attempts.append({"ORDER": (p, d, q), "AICC": fit.aicc})
        if best is None or fit.aicc < best[0].aicc:
            best = (fit, (p, d, q))
    return best, attempts


# Format the fitted ARIMA(p,d,q) equation with its estimated coefficients
def format_arima_equation(fit, order):
    p, d, q = order
    params = fit.params
    y_expr = "y'_t" if d else "y_t"

    def lag_label(i):
        return f"y'_{{t-{i}}}" if d else f"y_{{t-{i}}}"

    terms = []
    if "intercept" in params.index:
        terms.append(f"{params['intercept']:.3f}")
    for i in range(1, p + 1):
        coef = params[f"ar.L{i}"]
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.3f}*{lag_label(i)}")
    rhs = " ".join(terms) + " + e_t" if terms else "e_t"
    for j in range(1, q + 1):
        coef = params[f"ma.L{j}"]
        sign = "+" if coef >= 0 else "-"
        rhs += f" {sign} {abs(coef):.3f}*e_{{t-{j}}}"
    return f"{y_expr} = {rhs}"


def style_axis(ax, labelsize=7):
    ax.grid(axis="y", color=COLOR["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR["baseline"])
    ax.tick_params(colors=COLOR["muted"], labelsize=labelsize)


# League-wide 3PA/game per season
def analyze_league_trend():
    league = pd.read_csv(TS_FOLDER / "league_3pa_per_season.csv", index_col=0)["LEAGUE_3PA_PER_GAME"]
    league.index = range(len(league))

    result, attempts = fit_best_order(league)
    if result is None:
        print("League trend: no ARIMA model converged")
        return
    fit, order = result

    forecast = fit.get_forecast(steps=FORECAST_SEASONS)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05)

    future_seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2025, 2025 + FORECAST_SEASONS)]
    all_labels = SEASONS + future_seasons
    x_hist = range(len(SEASONS))
    x_fcst = range(len(SEASONS), len(SEASONS) + FORECAST_SEASONS)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(COLOR["surface"])
    ax.set_facecolor(COLOR["surface"])

    # The first `loglikelihood_burn` fitted values are an unreliable Kalman-filter warm-up artifact
    burn = fit.loglikelihood_burn
    ax.plot(x_hist, league.values, color=COLOR["blue"], linewidth=2, marker="o", markersize=4, label="Actual")
    ax.plot(list(x_hist)[burn:], fit.fittedvalues[burn:], color=COLOR["secondary_ink"], linewidth=1.3, linestyle="--", label="In-sample fit")
    ax.plot(x_fcst, forecast_mean, color=COLOR["highlight"], linewidth=2, marker="o", markersize=4, label=f"Forecast ({FORECAST_SEASONS} seasons)")
    ax.fill_between(x_fcst, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color=COLOR["highlight"], alpha=0.2, label="95% CI")

    ax.set_xticks(list(x_hist) + list(x_fcst))
    ax.set_xticklabels([label[-5:] for label in all_labels], rotation=45, ha="right", fontsize=8, color=COLOR["secondary_ink"])
    ax.set_ylabel("League-wide 3PA per game", color=COLOR["primary_ink"])
    ax.set_title(
        f"ARIMA({order[0]},{order[1]},{order[2]}) forecast – league-wide 3PA per game",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold", pad=14,
    )
    style_axis(ax, labelsize=9)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR["secondary_ink"])

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "league_3pa_arima.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Save a summary CSV with every attempted order and its AICc, plus the forecast for the best order
    for row in attempts:
        is_best = row["ORDER"] == order
        row["SERIES"] = "LEAGUE"
        row["BEST"] = is_best
        row["COEFFICIENTS"] = {name: {"COEF": round(float(fit.params[name]), 3), "SE": round(float(fit.bse[name]), 3), "P": round(float(fit.pvalues[name]), 4)} for name in fit.params.index} if is_best else None
        for s, v, lo, hi in zip(future_seasons, forecast_mean, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1]):
            row[f"FORECAST_{s}"] = v if is_best else None
            row[f"FORECAST_{s}_LOWER95"] = lo if is_best else None
            row[f"FORECAST_{s}_UPPER95"] = hi if is_best else None
    forecast_cols = [c for s in future_seasons for c in (f"FORECAST_{s}", f"FORECAST_{s}_LOWER95", f"FORECAST_{s}_UPPER95")]
    summary = pd.DataFrame(attempts)[
        ["SERIES", "ORDER", "AICC", "BEST", "COEFFICIENTS"] + forecast_cols
    ].sort_values("AICC", na_position="last")
    summary.to_csv(OUTPUT_FOLDER / "league_3pa_arima_summary.csv", index=False)
    print(f"League trend: ARIMA{order}, AICc {fit.aicc:.1f}")
    print(f"  {format_arima_equation(fit, order)}")


# Team-by-team 3PA/game per season
def analyze_team_season_trend():
    team_pivot = pd.read_csv(TS_FOLDER / "team_3pa_per_season.csv", index_col=0)
    teams = sorted(team_pivot.columns)

    future_seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2025, 2025 + FORECAST_SEASONS)]
    n_hist = len(SEASONS)
    all_labels = SEASONS + future_seasons
    tick_step = 3
    tick_idx = range(0, len(all_labels), tick_step)
    tick_pos = list(tick_idx)
    tick_labels = [all_labels[i][-5:] for i in tick_idx]

    ncols = 5
    nrows = -(-len(teams) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(COLOR["surface"])

    rows = []
    for ax, team in zip(axes.flat, teams):
        ax.set_facecolor(COLOR["surface"])
        series = team_pivot[team].dropna()
        series.index = range(len(series))
        color = TEAM_COLORS.get(team, COLOR["blue"])

        ax.plot(series.index, series.values, color=color, linewidth=1.3)

        result, _ = fit_best_order(series)
        if result is not None:
            fit, order = result
            forecast = fit.get_forecast(steps=FORECAST_SEASONS)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)
            fx = range(n_hist, n_hist + FORECAST_SEASONS)
            ax.plot(fx, forecast_mean, color=color, linewidth=1.3, linestyle="--")
            ax.fill_between(fx, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color=color, alpha=0.15)
            rows.append({
                "TEAM": team, "ORDER": order, "AICC": fit.aicc,
                **{f"FORECAST_{s}": v for s, v in zip(future_seasons, forecast_mean)},
            })
        else:
            rows.append({"TEAM": team, "ORDER": None, "AICC": None})

        ax.set_title(team, fontsize=8, fontweight="bold", color=COLOR["primary_ink"])
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6, color=COLOR["secondary_ink"])
        ax.label_outer()
        style_axis(ax)

    for ax in axes.flat[len(teams):]:
        ax.set_visible(False)

    fig.supxlabel("Season", color=COLOR["secondary_ink"], fontsize=10)
    fig.supylabel("3PA per game", color=COLOR["secondary_ink"], fontsize=9)
    fig.suptitle(
        f"Team-by-team 3PA/game per season – ARIMA fit + {FORECAST_SEASONS}-season forecast\n"
        "(dashed line/band: forecast and its 95% CI)",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0.015, 0.015, 1, 0.91))
    fig.savefig(OUTPUT_FOLDER / "team_3pa_arima.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    pd.DataFrame(rows).to_csv(OUTPUT_FOLDER / "team_3pa_arima_summary.csv", index=False)


# Team-by-team 3PA/game by point in the season
def _analyze_team_matchweek(csv_name, ylabel, title, out_name):
    matchweek_pivot = pd.read_csv(TS_FOLDER / csv_name, index_col=0)
    teams = sorted(matchweek_pivot.columns)

    ncols = 5
    nrows = -(-len(teams) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(COLOR["surface"])

    rows = []
    for ax, abbr in zip(axes.flat, teams):
        ax.set_facecolor(COLOR["surface"])
        series = matchweek_pivot[abbr].dropna()
        team_name = ABBR_TO_NAME.get(abbr, abbr)
        color = TEAM_COLORS.get(team_name, COLOR["blue"])

        ax.plot(series.index, series.values, color=color, linewidth=0.8, alpha=0.4)

        result, _ = fit_best_order(series)
        if result is not None:
            fit, order = result
            # The first `loglikelihood_burn` fitted values are an unreliable Kalman-filter warm-up artifact
            burn = fit.loglikelihood_burn
            ax.plot(series.index[burn:], fit.fittedvalues.values[burn:], color=color, linewidth=1.6)
            rows.append({"TEAM": team_name, "ORDER": order, "AICC": fit.aicc})
        else:
            rows.append({"TEAM": team_name, "ORDER": None, "AICC": None})

        ax.set_title(team_name, fontsize=8, fontweight="bold", color=COLOR["primary_ink"])
        style_axis(ax, labelsize=6)

    for ax in axes.flat[len(teams):]:
        ax.set_visible(False)

    fig.supxlabel("Game number in season", color=COLOR["secondary_ink"], fontsize=10)
    fig.supylabel(ylabel, color=COLOR["secondary_ink"], fontsize=9)
    fig.suptitle(
        f"{title} – ARIMA in-sample fit (bold) vs actual (faint)",
        color=COLOR["primary_ink"], fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0.015, 0.015, 1, 0.92))
    fig.savefig(OUTPUT_FOLDER / f"{out_name}.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    pd.DataFrame(rows).to_csv(OUTPUT_FOLDER / f"{out_name}_summary.csv", index=False)


def analyze_team_matchweek_trend():
    _analyze_team_matchweek(
        "team_3pa_per_game_number.csv",
        ylabel="3PA per game\n(avg across seasons at this game number)",
        title="Team 3PA per game by point in the season",
        out_name="team_3pa_matchweek_arima",
    )


def analyze_team_matchweek_relative_trend():
    _analyze_team_matchweek(
        "team_3pa_relative_per_game_number.csv",
        ylabel="3PA vs own season average (%)",
        title="Team 3PA relative to own season average, by point in the season",
        out_name="team_3pa_matchweek_relative_arima",
    )


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    analyze_league_trend()
    analyze_team_season_trend()
    analyze_team_matchweek_trend()
    analyze_team_matchweek_relative_trend()

    print("Saved outputs to", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
