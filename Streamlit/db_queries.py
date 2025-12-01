import sqlite3
import pandas as pd
import streamlit as st

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


@st.cache_data
def get_shots_with_xg(match_id):
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
            shot_technique
        FROM events
        WHERE match_id = {match_id}
        AND type = 'Shot'
        AND shot_statsbomb_xg IS NOT NULL;
    """
    df = pd.read_sql_query(query, conn)
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

    for team in [home_team, away_team]:
        query = f"""
            SELECT 
                COUNT(CASE WHEN type = 'Pass' THEN 1 END) as passes,
                COUNT(CASE WHEN type = 'Shot' THEN 1 END) as shots,
                COUNT(CASE WHEN type = 'Shot' AND shot_outcome = 'Goal' THEN 1 END) as goals,
                ROUND(SUM(CASE WHEN type = 'Shot' THEN shot_statsbomb_xg END), 3) as total_xg,
                COUNT(CASE WHEN type = 'Pass' AND pass_outcome IS NULL THEN 1 END) * 100.0 /
                    NULLIF(COUNT(CASE WHEN type = 'Pass' THEN 1 END), 0) as pass_accuracy,
                COUNT(CASE WHEN type = 'Duel' THEN 1 END) as duels,
                COUNT(CASE WHEN type = 'Foul Committed' THEN 1 END) as fouls
            FROM events
            WHERE match_id = {match_id} AND team = '{team}';
        """
        df = pd.read_sql_query(query, conn)
        stats[team] = df.iloc[0].to_dict()

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
        stats_query = f"""
            WITH team_events AS (
                SELECT * FROM events
                WHERE match_id IN ({match_ids_str})
                AND team = '{team}'
            )
            SELECT 
                COUNT(*) as total_events,
                COUNT(CASE WHEN type = 'Pass' THEN 1 END) as total_passes,
                COUNT(CASE WHEN type = 'Shot' THEN 1 END) as total_shots,
                COUNT(CASE WHEN type = 'Shot' AND shot_outcome = 'Goal' THEN 1 END) as goals_scored,
                COUNT(CASE WHEN type = 'Pass' AND pass_assisted_shot_id IS NOT NULL THEN 1 END) as assists,
                COUNT(CASE WHEN type = 'Duel' AND duel_outcome = 'Won' THEN 1 END) as duels_won,
                COUNT(CASE WHEN type = 'Interception' THEN 1 END) as interceptions,
                COUNT(CASE WHEN type = 'Foul Committed' THEN 1 END) as fouls_committed,
                COUNT(CASE WHEN type = 'Foul Won' THEN 1 END) as fouls_won,
                ROUND(SUM(CASE WHEN type = 'Shot' THEN shot_statsbomb_xg END), 3) as total_xg,
                ROUND(AVG(CASE WHEN type = 'Shot' THEN shot_statsbomb_xg END), 3) as avg_shot_xg,
                COUNT(CASE WHEN type = 'Pass' AND pass_outcome IS NULL THEN 1 END) * 100.0 /
                    NULLIF(COUNT(CASE WHEN type = 'Pass' THEN 1 END), 0) as pass_accuracy
            FROM team_events;
        """
        stats_df = pd.read_sql_query(stats_query, conn)

        if not stats_df.empty:
            stats = stats_df.iloc[0].to_dict()
            stats['matches_played'] = len(match_ids)
            stats['scoring_accuracy'] = (
                stats['goals_scored'] / stats['total_shots'] * 100 if stats['total_shots'] > 0 else 0
            )
            stats['xg_efficiency'] = (
                stats['goals_scored'] / stats['total_xg'] if stats['total_xg'] > 0 else 0
            )
            stats['goals_per_match'] = (
                stats['goals_scored'] / stats['matches_played'] if stats['matches_played'] > 0 else 0
            )
            comparison_data[team] = stats

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
