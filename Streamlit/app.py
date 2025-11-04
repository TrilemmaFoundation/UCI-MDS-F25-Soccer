
    # ------------------------------
    # Add scoreboard to 5 min intervals to get a better analysis
    # Map imgs with country flags
    # Ask about xG numbers
    # Add simple statistics







import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from aggregate_stats import get_team_aggregate_stats  # ✅ import existing function

# ------------------------------
# 1. Database Connection
# ------------------------------
DB_PATH = "/Users/indrajeet/Documents/Compsci 296p/UCI-MDS-F25-Soccer/statsbomb_euro2020.db"
# DB_PATH = r"C:\Users\Bryan\CodeProjects\D296P-297P\UCI-MDS-F25-Soccer\statsbomb_euro2020.db"
FLAGS_PATH = "/Users/indrajeet/Documents/Compsci 296p/UCI-MDS-F25-Soccer/flags"  # Folder with flag images

# ------------------------------
# Cached Database Queries
# ------------------------------
@st.cache_data
def get_teams():
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
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT match_id, home_team, away_team, match_date, home_score, away_score
        FROM matches
        WHERE home_team = '{team}' OR away_team = '{team}';
    """
    matches = pd.read_sql_query(query, conn)
    conn.close()
    return matches

@st.cache_data
def get_events(match_id):
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM events WHERE match_id = {match_id};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ------------------------------
# Passes per 5-minute intervals
# ------------------------------
def get_pass_intervals(match_id):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        WITH pass_intervals AS (
            SELECT
                team,
                ((minute / 5) * 5) AS interval_start,
                COUNT(*) AS pass_count
            FROM passes
            WHERE match_id = {match_id}
            GROUP BY team, interval_start
        )
        SELECT * FROM pass_intervals
        ORDER BY interval_start, team;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ------------------------------
# 2. Sidebar UI
# ------------------------------
st.sidebar.title("⚽ Soccer Analytics Dashboard")
selected_team = st.sidebar.selectbox("Select a Team", get_teams())

if selected_team:
    matches = get_matches(selected_team)
    match_options = [
        f"{row['home_team']} vs {row['away_team']} ({row['match_date']})"
        for _, row in matches.iterrows()
    ]
    match_options.insert(0, "🔹 View Match Stats")

    selected_match = st.sidebar.selectbox("Select a Match", match_options)

    if selected_match != "🔹 View Match Stats":
        match_row = matches.iloc[match_options.index(selected_match) - 1]
        match_id = match_row["match_id"]
        home_team = match_row["home_team"]
        away_team = match_row["away_team"]
        home_score = match_row["home_score"]
        away_score = match_row["away_score"]

        st.sidebar.markdown("---")
        st.sidebar.write(f"**Home Team:** {home_team}")
        st.sidebar.write(f"**Away Team:** {away_team}")
        st.sidebar.write(f"**Match Score:** {home_score} - {away_score}")
        st.sidebar.write(f"**Match ID:** {match_id}")

# ------------------------------
# 3. Main View
# ------------------------------
st.title("📊 Match Analytics Dashboard")

# ------------------------------
# 3A. Aggregate Stats View
# ------------------------------
if selected_team and selected_match == "🔹 View Match Stats":
    st.subheader(f"📈 Aggregate Statistics for {selected_team}")
    df_agg = get_team_aggregate_stats(DB_PATH, selected_team)

    if df_agg is None or df_agg.empty:
        st.warning("No aggregate data found for this team.")
    else:
        st.dataframe(df_agg)

# ------------------------------
# 3B. Match View (Score + Graph + Heatmaps)
# ------------------------------
elif selected_team and selected_match and selected_match != "🔹 View Match Stats":
    st.subheader(f"🏟️ {selected_match}")

    # Display score prominently
    st.markdown(f"### ⚽ Final Score: **{home_team} {home_score} - {away_score} {away_team}**")

    # ------------------------------
    # Passes per 5-Minute Interval Chart
    # ------------------------------
    st.markdown("#### 📊 Passes per 5-Minute Interval by Team")
    df_pass = get_pass_intervals(match_id)

    if not df_pass.empty:
        pivot_df = df_pass.pivot(index="interval_start", columns="team", values="pass_count").fillna(0)

        fig, ax = plt.subplots(figsize=(8, 5))
        for team in pivot_df.columns:
            ax.plot(pivot_df.index, pivot_df[team], marker='o', label=team)
        ax.set_title("Passes per 5-Minute Interval by Team")
        ax.set_xlabel("Time Interval (minutes)")
        ax.set_ylabel("Number of Passes")
        ax.legend(title="Team")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
    else:
        st.warning("No passing data found for this match.")

    # ------------------------------
    # Event Heatmaps
    # ------------------------------
    df_events = get_events(match_id)

    def display_team_header(team_name):
        col_flag, col_text = st.columns([1, 5])
        flag_path = os.path.join(FLAGS_PATH, f"{team_name}.png")
        with col_flag:
            if os.path.exists(flag_path):
                st.image(flag_path, width=50)
            else:
                st.write("🏳️")
        with col_text:
            st.markdown(f"### **{team_name}**")

    def plot_heatmap(df, event_type, title):
        event_df = df[df["type"].str.contains(event_type, case=False, na=False)]

        if "location" in event_df.columns:
            event_df["x"] = event_df["location"].apply(
                lambda s: float(s.strip("[]").split(",")[0]) if pd.notnull(s) else np.nan
            )
            event_df["y"] = event_df["location"].apply(
                lambda s: float(s.strip("[]").split(",")[1]) if pd.notnull(s) else np.nan
            )
        else:
            event_df["x"] = np.random.uniform(0, 120, 100)
            event_df["y"] = np.random.uniform(0, 80, 100)

        heatmap, _, _ = np.histogram2d(event_df["x"], event_df["y"], bins=50, range=[[0, 120], [0, 80]])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(heatmap.T, extent=[0, 120, 0, 80], origin="lower", cmap="turbo", alpha=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        st.pyplot(fig)

    if "team" in df_events.columns:
        st.markdown("## 🎯 Comparative Event Heatmaps")
        event_types = ["Shot", "Pass", "Ball Receipt"]
        event_titles = ["Shot Heatmap", "Pass Heatmap", "Ball Receipt Heatmap"]

        for event_type, title in zip(event_types, event_titles):
            st.markdown(f"### ⚔️ {title}")
            col1, col2 = st.columns(2)
            with col1:
                display_team_header(home_team)
                plot_heatmap(df_events[df_events["team"] == home_team], event_type, home_team)
            with col2:
                display_team_header(away_team)
                plot_heatmap(df_events[df_events["team"] == away_team], event_type, away_team)
