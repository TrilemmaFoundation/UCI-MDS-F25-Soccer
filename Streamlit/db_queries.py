import sqlite3
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
import ast
from metrics.xg import load_model, XG_FEATURES
from metrics.data_processing import (
    compute_distance_vec,
    compute_angle_vec,
    count_players,
    safe_json_list,
)

pipeline: Pipeline = load_model("../metrics/weights/xg_pipeline.pkl")

# Global DB path
DB_PATH = "../statsbomb/statsbomb_euro2020.db"


# -----------------------------------
# Basic Queries
# -----------------------------------

@st.cache_data
def get_teams():
    """Return all distinct teams from matches."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT DISTINCT home_team AS team FROM matches
        UNION
        SELECT DISTINCT away_team AS team FROM matches
        ORDER BY team;
    """
    teams = pd.read_sql_query(query, conn)
    conn.close()
    return teams["team"].tolist()


@st.cache_data
def get_matches(team):
    """Return all matches played by a specific team."""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT match_id, home_team, away_team, match_date, home_score, away_score, competition_stage
        FROM matches
        WHERE home_team = '{team}' OR away_team = '{team}'
        ORDER BY match_date;
    """
    matches = pd.read_sql_query(query, conn)
    conn.close()
    return matches


@st.cache_data
def get_events(match_id):
    """Return all events for a given match."""
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM events WHERE match_id = {match_id};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def preprocess_shots(df):
    df[["location_x", "location_y"]] = (
        df["location"].apply(lambda s: ast.literal_eval(s)).apply(pd.Series)
    )
    df = df.assign(
        distance=lambda d: compute_distance_vec(d["location_x"], d["location_y"]),
        angle=lambda d: compute_angle_vec(d["location_x"], d["location_y"]),
    )
    df["freeze_list"] = df["shot_freeze_frame"].apply(safe_json_list)
    df[["num_teammates", "num_opponents"]] = (
        df["freeze_list"].apply(count_players).to_list()
    )
    return df


@st.cache_data
def get_shots_with_xg(match_id, use_custom_xg=True):
    """Return all shots with xG values for a given match."""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT 
            team,
            location,
            shot_statsbomb_xg,
            player,
            minute,
            second,
            shot_outcome,
            shot_body_part,
            shot_technique,
            shot_type,
            play_pattern,
            shot_freeze_frame
        FROM events
        WHERE match_id = {match_id} 
        AND type = 'Shot';
    """
    df = pd.read_sql_query(query, conn)
    df = preprocess_shots(df)
    df["xg"] = pipeline.predict_proba(df[XG_FEATURES])[:, 1]

    conn.close()
    return df


# -----------------------------------
# Match-Level Stats
# -----------------------------------

@st.cache_data
def get_match_summary_stats(match_id, home_team, away_team):
    """Compute basic match summary stats (shots, passes, goals, xG, etc.)."""
    conn = sqlite3.connect(DB_PATH)
    stats = {}

    query = f"""
        SELECT 
            team,
            location,
            shot_body_part,
            shot_technique,
            shot_type,
            play_pattern,
            shot_freeze_frame
        FROM events
        WHERE match_id = {match_id} 
        AND type = 'Shot';
    """
    shots_df = pd.read_sql_query(query, conn)
    shots_df = preprocess_shots(shots_df)
    shots_df["xg"] = pipeline.predict_proba(shots_df[XG_FEATURES])[:, 1]
    model_xg = shots_df.groupby("team")["xg"].sum().round(3)

    for team in [home_team, away_team]:
        query = f"""
            SELECT 
                COUNT(CASE WHEN type = 'Pass' THEN 1 END) as passes,
                COUNT(CASE WHEN type = 'Shot' THEN 1 END) as shots,
                COUNT(CASE WHEN type = 'Shot' AND shot_outcome = 'Goal' THEN 1 END) as goals,
                ROUND(SUM(CASE WHEN type = 'Shot' THEN shot_statsbomb_xg END), 3) as statsbomb_total_xg,
                COUNT(CASE WHEN type = 'Pass' AND pass_outcome IS NULL THEN 1 END) * 100.0 /
                    NULLIF(COUNT(CASE WHEN type = 'Pass' THEN 1 END), 0) as pass_accuracy,
                COUNT(CASE WHEN type = 'Duel' THEN 1 END) as duels,
                COUNT(CASE WHEN type = 'Foul Committed' THEN 1 END) as fouls
            FROM events
            WHERE match_id = {match_id} AND team = '{team}';
        """
        df = pd.read_sql_query(query, conn)
        row = df.iloc[0].to_dict()
        row["total_xg"] = float(model_xg.get(team, 0.0))
        stats[team] = row

    conn.close()
    return stats


# -----------------------------------
# Interval-Based Queries (5-Minute)
# -----------------------------------

def _interval_query(match_id, event_type, count_label):
    """Generic helper for event count over 5-minute intervals."""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        WITH intervals AS (
            SELECT
                team,
                ((minute / 5) * 5) AS interval_start,
                COUNT(*) AS {count_label}
            FROM events
            WHERE match_id = {match_id}
            AND type = '{event_type}'
            GROUP BY team, interval_start
        )
        SELECT * FROM intervals
        ORDER BY interval_start, team;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


@st.cache_data
def get_pass_intervals(match_id):
    return _interval_query(match_id, "Pass", "pass_count")


@st.cache_data
def get_shot_intervals(match_id):
    return _interval_query(match_id, "Shot", "shot_count")


@st.cache_data
def get_receive_intervals(match_id):
    """Return number of Ball Receipts by 5-minute interval."""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        WITH receive_intervals AS (
            SELECT
                team,
                ((minute / 5) * 5) AS interval_start,
                COUNT(*) AS receive_count
            FROM events
            WHERE match_id = {match_id}
            AND type LIKE 'Ball Receipt%'
            GROUP BY team, interval_start
        )
        SELECT * FROM receive_intervals
        ORDER BY interval_start, team;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# -----------------------------------
# Other Match Queries
# -----------------------------------

@st.cache_data
def get_passes_by_interval(match_id, start_minute, end_minute):
    """Return pass events for a specific time interval."""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT team, location, minute, player
        FROM events
        WHERE match_id = {match_id}
        AND type = 'Pass'
        AND location IS NOT NULL
        AND minute >= {start_minute}
        AND minute < {end_minute}
        ORDER BY minute, second;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


@st.cache_data
def get_match_duration(match_id):
    """Return maximum minute (rounded to nearest 5) for a match."""
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT MAX(minute) AS max_minute FROM events WHERE match_id = {match_id};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    max_min = df.iloc[0]['max_minute']
    return int(((max_min // 5) + 1) * 5) if max_min else 90


# -----------------------------------
# Aggregate Team Stats
# -----------------------------------

@st.cache_data
def get_team_comparison_data():
    """Return aggregate stats for all teams across all matches."""
    conn = sqlite3.connect(DB_PATH)
    teams_query = """
        SELECT DISTINCT home_team AS team FROM matches
        UNION
        SELECT DISTINCT away_team AS team FROM matches
        ORDER BY team;
    """
    teams = pd.read_sql_query(teams_query, conn)["team"].tolist()
    comparison_data = {}

    for team in teams:
        match_ids_query = f"""
            SELECT match_id FROM matches
            WHERE home_team = '{team}' OR away_team = '{team}';
        """
        match_ids_df = pd.read_sql_query(match_ids_query, conn)
        match_ids = match_ids_df["match_id"].tolist()
        if not match_ids:
            continue

        match_ids_str = ",".join(map(str, match_ids))

        # all event rows for this team
        events_query = f"""
            SELECT * FROM events
            WHERE match_id IN ({match_ids_str})
            AND team = '{team}';
        """
        events_df = pd.read_sql_query(events_query, conn)

        total_events = len(events_df)
        total_passes = (events_df["type"] == "Pass").sum()
        total_shots = (events_df["type"] == "Shot").sum()
        goals_scored = (
            (events_df["type"] == "Shot") & (events_df["shot_outcome"] == "Goal")
        ).sum()
        assists = (
            (events_df["type"] == "Pass") & events_df["pass_assisted_shot_id"].notna()
        ).sum()
        duels_won = (
            (events_df["type"] == "Duel") & (events_df["duel_outcome"] == "Won")
        ).sum()
        interceptions = (events_df["type"] == "Interception").sum()
        fouls_committed = (events_df["type"] == "Foul Committed").sum()
        fouls_won = (events_df["type"] == "Foul Won").sum()

        # compute model xG using model
        shots_df = events_df[events_df["type"] == "Shot"].copy()
        shots_df = preprocess_shots(shots_df)
        shots_df["xg"] = pipeline.predict_proba(shots_df[XG_FEATURES])[:, 1]
        total_xg = shots_df["xg"].sum()
        avg_shot_xg = shots_df["xg"].mean() if not shots_df["xg"].empty else 0

        matches_played = len(match_ids)
        scoring_accuracy = goals_scored / total_shots * 100 if total_shots > 0 else 0
        xg_efficiency = (goals_scored / total_xg) if total_xg > 0 else 0
        goals_per_match = goals_scored / matches_played if matches_played > 0 else 0
        pass_accuracy = (
            (events_df["type"].eq("Pass") & events_df["pass_outcome"].isna()).sum()
            * 100.0
            / total_passes
            if total_passes > 0
            else 0
        )

        comparison_data[team] = {
            "total_events": total_events,
            "total_passes": total_passes,
            "total_shots": total_shots,
            "goals_scored": goals_scored,
            "assists": assists,
            "duels_won": duels_won,
            "interceptions": interceptions,
            "fouls_committed": fouls_committed,
            "fouls_won": fouls_won,
            "total_xg": round(total_xg, 3),
            "avg_shot_xg": round(avg_shot_xg, 3),
            "pass_accuracy": pass_accuracy,
            "matches_played": matches_played,
            "scoring_accuracy": scoring_accuracy,
            "xg_efficiency": xg_efficiency,
            "goals_per_match": goals_per_match,
        }

    conn.close()
    return comparison_data


@st.cache_data
def get_team_ranking_data(comparison_data, metric, ascending=False, min_matches=1):
    """Return ranked team data for a given metric."""
    ranked_data = []
    for team, stats in comparison_data.items():
        if stats['matches_played'] >= min_matches and metric in stats:
            ranked_data.append({
                'team': team,
                'value': stats[metric],
                'matches_played': stats['matches_played']
            })

    ranked_df = pd.DataFrame(ranked_data)
    if not ranked_df.empty:
        ranked_df = ranked_df.sort_values('value', ascending=ascending)
    return ranked_df
