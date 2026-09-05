import time
from pathlib import Path

from nba_api.stats.endpoints import (
    leaguedashteamstats,
    leaguegamelog,
    playercareerstats,
)
from nba_api.stats.static import players



# All 21 seasons
SEASONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2004, 2025)]

# Only these 3 seasons are used for the spatial (KDE) analysis
# SPATIAL_SEASONS = ["2004-05", "2014-15", "2024-25"]

# Players to collect shot data for (Curry + comparison players)
PLAYER_NAMES = ["Stephen Curry", "Klay Thompson", "James Harden", "Damian Lillard", "Kevin Durant"]

# Wait between API calls so we don't overload the NBA API
DELAY = 3.0
MAX_RETRIES = 5

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data"



def save_csv(df, filepath):
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath} ({len(df)} rows)")


def already_downloaded(filepath):
    if filepath.exists():
        print(f"Skipping {filepath} (already downloaded)")
        return True
    return False


def get_player_id(name):
    result = players.find_players_by_full_name(name)
    return result[0]["id"]


# Retry API calls on connection errors
def fetch_with_retry(fn):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = DELAY * (2 ** attempt)
            print(f"  Connection error (attempt {attempt}/{MAX_RETRIES}): {e}")
            print(f"  Retrying in {wait:.0f}s...")
            time.sleep(wait)



# Team stats for every season
def collect_team_stats():
    print("Fetching Team stats per season")
    for season in SEASONS:
        filepath = DATA_FOLDER / "team_stats" / f"team_stats_{season}.csv"
        if already_downloaded(filepath):
            continue

        time.sleep(DELAY)
        stats = fetch_with_retry(lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        ))
        df = stats.get_data_frames()[0]
        df["SEASON"] = season
        save_csv(df, filepath)


# Game logs for every season
def collect_game_logs():
    print("\nFetching Game logs per season")
    for season in SEASONS:
        filepath = DATA_FOLDER / "game_logs" / f"game_log_{season}.csv"
        if already_downloaded(filepath):
            continue

        time.sleep(DELAY)
        log = fetch_with_retry(lambda: leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
        ))
        df = log.get_data_frames()[0]
        df["SEASON"] = season
        save_csv(df, filepath)



# Career stats for each player
def collect_player_stats():
    print("\nFetching Player career stats ---")
    for name in PLAYER_NAMES:
        filename = name.lower().replace(" ", "_")
        filepath = DATA_FOLDER / "players" / f"{filename}_career.csv"
        if already_downloaded(filepath):
            continue

        time.sleep(DELAY)
        player_id = get_player_id(name)
        career = fetch_with_retry(lambda: playercareerstats.PlayerCareerStats(player_id=player_id))
        df = career.get_data_frames()[0]
        df = df[df["SEASON_ID"] <= "2024-25"]
        save_csv(df, filepath)


if __name__ == "__main__":
    for folder in [DATA_FOLDER / "team_stats", DATA_FOLDER / "game_logs",
                   DATA_FOLDER / "players"]:
        folder.mkdir(parents=True, exist_ok=True)

    collect_team_stats()
    collect_game_logs()
    collect_player_stats()

    print("\nData collected and saved.")
