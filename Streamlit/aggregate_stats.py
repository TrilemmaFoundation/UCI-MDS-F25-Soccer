import sqlite3
import pandas as pd

DB_PATH = "../statsbomb/statsbomb_euro2020.db"

# def get_teams():
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql_query("SELECT DISTINCT team FROM events ORDER BY team;", conn)
#     conn.close()
#     return df['team'].tolist()

def get_team_aggregate_stats(DB_PATH, team_name):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        WITH team_stats AS (
            SELECT
                team,
                COUNT(*) AS total_events,
                SUM(CASE WHEN type = 'Pass' THEN 1 ELSE 0 END) AS total_passes,
                SUM(CASE WHEN type = 'Shot' THEN 1 ELSE 0 END) AS total_shots,
                SUM(CASE WHEN shot_outcome = 'Goal' THEN 1 ELSE 0 END) AS total_goals,
                ROUND(AVG(shot_statsbomb_xg), 3) AS avg_xg
            FROM events
            WHERE team = '{team_name}'
            GROUP BY team
        )
        SELECT * FROM team_stats;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# print(get_teams())
# print(get_team_aggregate_stats('Austria'))
