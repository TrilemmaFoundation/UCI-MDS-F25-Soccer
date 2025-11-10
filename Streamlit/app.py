
#     # ------------------------------
#     # Add scoreboard to 5 min intervals to get a better analysis
#     # Map imgs with country flags
#     # Ask about xG numbers
#     # Add simple statistics







# import streamlit as st
# import sqlite3
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from aggregate_stats import get_team_aggregate_stats  # ✅ import existing function

# # ------------------------------
# # 1. Database Connection
# # ------------------------------
# DB_PATH = "/Users/indrajeet/Documents/Compsci 296p/UCI-MDS-F25-Soccer/statsbomb_euro2020.db"
# # DB_PATH = r"C:\Users\Bryan\CodeProjects\D296P-297P\UCI-MDS-F25-Soccer\statsbomb_euro2020.db"
# FLAGS_PATH = "/Users/indrajeet/Documents/Compsci 296p/UCI-MDS-F25-Soccer/flags"  # Folder with flag images

# # ------------------------------
# # Cached Database Queries
# # ------------------------------
# @st.cache_data
# def get_teams():
#     conn = sqlite3.connect(DB_PATH)
#     query = """
#         SELECT DISTINCT home_team AS team FROM matches
#         UNION
#         SELECT DISTINCT away_team AS team FROM matches
#         ORDER BY team;
#     """
#     teams = pd.read_sql_query(query, conn)
#     conn.close()
#     return teams["team"].tolist()

# @st.cache_data
# def get_matches(team):
#     conn = sqlite3.connect(DB_PATH)
#     query = f"""
#         SELECT match_id, home_team, away_team, match_date, home_score, away_score
#         FROM matches
#         WHERE home_team = '{team}' OR away_team = '{team}';
#     """
#     matches = pd.read_sql_query(query, conn)
#     conn.close()
#     return matches

# @st.cache_data
# def get_events(match_id):
#     conn = sqlite3.connect(DB_PATH)
#     query = f"SELECT * FROM events WHERE match_id = {match_id};"
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#     return df

# # ------------------------------
# # Passes per 5-minute intervals
# # ------------------------------
# def get_pass_intervals(match_id):
#     conn = sqlite3.connect(DB_PATH)
#     query = f"""
#         WITH pass_intervals AS (
#             SELECT
#                 team,
#                 ((minute / 5) * 5) AS interval_start,
#                 COUNT(*) AS pass_count
#             FROM passes
#             WHERE match_id = {match_id}
#             GROUP BY team, interval_start
#         )
#         SELECT * FROM pass_intervals
#         ORDER BY interval_start, team;
#     """
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#     return df

# # ------------------------------
# # 2. Sidebar UI
# # ------------------------------
# st.sidebar.title("⚽ Soccer Analytics Dashboard")
# selected_team = st.sidebar.selectbox("Select a Team", get_teams())

# if selected_team:
#     matches = get_matches(selected_team)
#     match_options = [
#         f"{row['home_team']} vs {row['away_team']} ({row['match_date']})"
#         for _, row in matches.iterrows()
#     ]
#     match_options.insert(0, "🔹 View Match Stats")

#     selected_match = st.sidebar.selectbox("Select a Match", match_options)

#     if selected_match != "🔹 View Match Stats":
#         match_row = matches.iloc[match_options.index(selected_match) - 1]
#         match_id = match_row["match_id"]
#         home_team = match_row["home_team"]
#         away_team = match_row["away_team"]
#         home_score = match_row["home_score"]
#         away_score = match_row["away_score"]

#         st.sidebar.markdown("---")
#         st.sidebar.write(f"**Home Team:** {home_team}")
#         st.sidebar.write(f"**Away Team:** {away_team}")
#         st.sidebar.write(f"**Match Score:** {home_score} - {away_score}")
#         st.sidebar.write(f"**Match ID:** {match_id}")

# # ------------------------------
# # 3. Main View
# # ------------------------------
# st.title("📊 Match Analytics Dashboard")

# # ------------------------------
# # 3A. Aggregate Stats View
# # ------------------------------
# if selected_team and selected_match == "🔹 View Match Stats":
#     st.subheader(f"📈 Aggregate Statistics for {selected_team}")
#     df_agg = get_team_aggregate_stats(DB_PATH, selected_team)

#     if df_agg is None or df_agg.empty:
#         st.warning("No aggregate data found for this team.")
#     else:
#         st.dataframe(df_agg)

# # ------------------------------
# # 3B. Match View (Score + Graph + Heatmaps)
# # ------------------------------
# elif selected_team and selected_match and selected_match != "🔹 View Match Stats":
#     st.subheader(f"🏟️ {selected_match}")

#     # Display score prominently
#     st.markdown(f"### ⚽ Final Score: **{home_team} {home_score} - {away_score} {away_team}**")

#     # ------------------------------
#     # Passes per 5-Minute Interval Chart
#     # ------------------------------
#     st.markdown("#### 📊 Passes per 5-Minute Interval by Team")
#     df_pass = get_pass_intervals(match_id)

#     if not df_pass.empty:
#         pivot_df = df_pass.pivot(index="interval_start", columns="team", values="pass_count").fillna(0)

#         fig, ax = plt.subplots(figsize=(8, 5))
#         for team in pivot_df.columns:
#             ax.plot(pivot_df.index, pivot_df[team], marker='o', label=team)
#         ax.set_title("Passes per 5-Minute Interval by Team")
#         ax.set_xlabel("Time Interval (minutes)")
#         ax.set_ylabel("Number of Passes")
#         ax.legend(title="Team")
#         ax.grid(True, linestyle='--', alpha=0.5)
#         st.pyplot(fig)
#     else:
#         st.warning("No passing data found for this match.")

#     # ------------------------------
#     # Event Heatmaps
#     # ------------------------------
#     df_events = get_events(match_id)

#     def display_team_header(team_name):
#         col_flag, col_text = st.columns([1, 5])
#         flag_path = os.path.join(FLAGS_PATH, f"{team_name}.png")
#         with col_flag:
#             if os.path.exists(flag_path):
#                 st.image(flag_path, width=50)
#             else:
#                 st.write("🏳️")
#         with col_text:
#             st.markdown(f"### **{team_name}**")

#     def plot_heatmap(df, event_type, title):
#         event_df = df[df["type"].str.contains(event_type, case=False, na=False)]

#         if "location" in event_df.columns:
#             event_df["x"] = event_df["location"].apply(
#                 lambda s: float(s.strip("[]").split(",")[0]) if pd.notnull(s) else np.nan
#             )
#             event_df["y"] = event_df["location"].apply(
#                 lambda s: float(s.strip("[]").split(",")[1]) if pd.notnull(s) else np.nan
#             )
#         else:
#             event_df["x"] = np.random.uniform(0, 120, 100)
#             event_df["y"] = np.random.uniform(0, 80, 100)

#         heatmap, _, _ = np.histogram2d(event_df["x"], event_df["y"], bins=50, range=[[0, 120], [0, 80]])
#         fig, ax = plt.subplots(figsize=(5, 4))
#         ax.imshow(heatmap.T, extent=[0, 120, 0, 80], origin="lower", cmap="turbo", alpha=0.6)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(title, fontsize=10)
#         st.pyplot(fig)

#     if "team" in df_events.columns:
#         st.markdown("## 🎯 Comparative Event Heatmaps")
#         event_types = ["Shot", "Pass", "Ball Receipt"]
#         event_titles = ["Shot Heatmap", "Pass Heatmap", "Ball Receipt Heatmap"]

#         for event_type, title in zip(event_types, event_titles):
#             st.markdown(f"### ⚔️ {title}")
#             col1, col2 = st.columns(2)
#             with col1:
#                 display_team_header(home_team)
#                 plot_heatmap(df_events[df_events["team"] == home_team], event_type, home_team)
#             with col2:
#                 display_team_header(away_team)
#                 plot_heatmap(df_events[df_events["team"] == away_team], event_type, away_team)






import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
from aggregate_stats import get_team_aggregate_stats
from PIL import Image

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Soccer Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# 1. Database Connection
# ------------------------------
DB_PATH = "/Users/indrajeet/Documents/Compsci 296p/UCI-MDS-F25-Soccer/statsbomb_euro2020.db"
FLAGS_PATH = "/Users/indrajeet/Documents/Compsci 296p/UCI-MDS-F25-Soccer/flags"
FIELD_IMAGE_PATH = "soccer-field.jpg"

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
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM events WHERE match_id = {match_id};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data
def get_shots_with_xg(match_id):
    """Get shot events with xG values"""
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

@st.cache_data
def get_match_summary_stats(match_id, home_team, away_team):
    """Get summary statistics for a match"""
    conn = sqlite3.connect(DB_PATH)
    
    stats = {}
    for team in [home_team, away_team]:
        query = f"""
            SELECT 
                COUNT(CASE WHEN type = 'Pass' THEN 1 END) as passes,
                COUNT(CASE WHEN type = 'Shot' THEN 1 END) as shots,
                COUNT(CASE WHEN type = 'Shot' AND shot_outcome = 'Goal' THEN 1 END) as goals,
                ROUND(AVG(CASE WHEN type = 'Shot' THEN shot_statsbomb_xg END), 3) as avg_xg,
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

# ------------------------------
# Event interval queries
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

def get_shot_intervals(match_id):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        WITH shot_intervals AS (
            SELECT
                team,
                ((minute / 5) * 5) AS interval_start,
                COUNT(*) AS shot_count
            FROM events
            WHERE match_id = {match_id}
            AND type = 'Shot'
            GROUP BY team, interval_start
        )
        SELECT * FROM shot_intervals
        ORDER BY interval_start, team;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_receive_intervals(match_id):
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

# ------------------------------
# Helper Functions
# ------------------------------
def display_team_header(team_name):
    """Display team name with flag"""
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
    """Matplotlib heatmap overlaid on soccer field image"""
    event_df = df[df["type"].str.contains(event_type, case=False, na=False)]

    if "location" in event_df.columns:
        event_df["x"] = event_df["location"].apply(
            lambda s: float(s.strip("[]").split(",")[0]) if pd.notnull(s) else np.nan
        )
        event_df["y"] = event_df["location"].apply(
            lambda s: float(s.strip("[]").split(",")[1]) if pd.notnull(s) else np.nan
        )
    else:
        st.warning(f"No {event_type} data available")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    
    if os.path.exists(FIELD_IMAGE_PATH):
        try:
            img = plt.imread(FIELD_IMAGE_PATH)
            ax.imshow(img, extent=[0, 120, 0, 80], aspect='auto', alpha=0.5)
        except Exception:
            pass
    
    heatmap, _, _ = np.histogram2d(event_df["x"], event_df["y"], bins=50, range=[[0, 120], [0, 80]])
    ax.imshow(heatmap.T, extent=[0, 120, 0, 80], origin="lower", cmap="turbo", alpha=0.6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold')
    st.pyplot(fig)
    plt.close()

def plot_interactive_shot_map(df_shots, team_name):
    """Interactive shot map with xG values on hover using Plotly"""
    team_shots = df_shots[df_shots["team"] == team_name].copy()
    
    if team_shots.empty:
        st.info(f"No shots recorded for {team_name}")
        return
    
    team_shots["x"] = team_shots["location"].apply(
        lambda s: float(s.strip("[]").split(",")[0]) if pd.notnull(s) else np.nan
    )
    team_shots["y"] = team_shots["location"].apply(
        lambda s: float(s.strip("[]").split(",")[1]) if pd.notnull(s) else np.nan
    )
    
    team_shots = team_shots.dropna(subset=["x", "y", "shot_statsbomb_xg"])
    
    if team_shots.empty:
        st.info(f"No valid shot data for {team_name}")
        return
    
    team_shots["hover_text"] = team_shots.apply(
        lambda row: f"<b>{row['player']}</b><br>"
                   f"xG: {row['shot_statsbomb_xg']:.3f}<br>"
                   f"Time: {int(row['minute'])}:{int(row['second']):02d}<br>"
                   f"Outcome: {row['shot_outcome']}<br>"
                   f"Body Part: {row['shot_body_part']}<br>"
                   f"Technique: {row['shot_technique']}",
        axis=1
    )
    
    fig = go.Figure()
    
    if os.path.exists(FIELD_IMAGE_PATH):
        try:
            img = Image.open(FIELD_IMAGE_PATH)
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x",
                    yref="y",
                    x=0,
                    y=80,
                    sizex=120,
                    sizey=80,
                    sizing="stretch",
                    opacity=0.5,
                    layer="below"
                )
            )
        except Exception:
            pass
    
    fig.add_trace(go.Scatter(
        x=team_shots["x"],
        y=team_shots["y"],
        mode='markers',
        marker=dict(
            size=12,
            color=team_shots["shot_statsbomb_xg"],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="xG", thickness=15),
            line=dict(width=1, color='black'),
            opacity=0.8
        ),
        text=team_shots["hover_text"],
        hovertemplate='%{text}<extra></extra>',
        name=team_name
    ))
    
    fig.update_layout(
        title=f"{team_name} - Shot Map",
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
        height=400,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_interval_chart(df, y_col, title, ylabel, color1='#1f77b4', color2='#ff7f0e'):
    """Create a clean interval chart"""
    if df.empty:
        st.warning(f"No data available for {title}")
        return
    
    pivot_df = df.pivot(index="interval_start", columns="team", values=y_col).fillna(0)
    
    fig = go.Figure()
    
    for i, team in enumerate(pivot_df.columns):
        color = color1 if i == 0 else color2
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[team],
            mode='lines+markers',
            name=team,
            line=dict(width=3, color=color),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Interval (minutes)",
        yaxis_title=ylabel,
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 2. Sidebar UI
# ------------------------------
with st.sidebar:
    st.markdown("# ⚽ Soccer Analytics")
    st.markdown("### UEFA Euro 2020")
    st.markdown("---")
    
    selected_team = st.selectbox("🏴 Select a Team", get_teams(), key="team_select")
    
    if selected_team:
        matches = get_matches(selected_team)
        match_options = [
            f"{row['home_team']} vs {row['away_team']}"
            for _, row in matches.iterrows()
        ]
        match_options.insert(0, "📊 Team Overview")
        
        selected_match = st.selectbox("⚽ Select a Match", match_options, key="match_select")
        
        if selected_match != "📊 Team Overview":
            match_row = matches.iloc[match_options.index(selected_match) - 1]
            match_id = match_row["match_id"]
            home_team = match_row["home_team"]
            away_team = match_row["away_team"]
            home_score = match_row["home_score"]
            away_score = match_row["away_score"]
            match_date = match_row["match_date"]
            stage = match_row["competition_stage"]
            
            st.markdown("---")
            st.markdown("### Match Info")
            st.write(f"**Date:** {match_date}")
            st.write(f"**Stage:** {stage}")
            st.write(f"**Score:** {home_score} - {away_score}")

# ------------------------------
# 3. Main View
# ------------------------------
st.markdown('<h1 class="main-header">⚽ Match Analytics Dashboard</h1>', unsafe_allow_html=True)

# ------------------------------
# 3A. Team Overview
# ------------------------------
if selected_team and selected_match == "📊 Team Overview":
    st.markdown(f"## {selected_team} - Tournament Statistics")
    
    # Team-level stats
    df_team_agg = get_team_aggregate_stats(DB_PATH, selected_team)
    
    if df_team_agg is not None and not df_team_agg.empty:
        st.markdown("### 📊 Team Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Passes", f"{df_team_agg['total_passes'].iloc[0]:,}")
        with col2:
            st.metric("Total Shots", f"{df_team_agg['total_shots'].iloc[0]}")
        with col3:
            st.metric("Goals Scored", f"{df_team_agg['total_goals'].iloc[0]}")
        with col4:
            st.metric("Avg xG", f"{df_team_agg['avg_xg'].iloc[0]:.3f}")
        with col5:
            st.metric("Total Events", f"{df_team_agg['total_events'].iloc[0]:,}")
        
        st.markdown("---")
    
    # Player-level stats
    st.markdown("### 👥 Player Performance")
    
    conn = sqlite3.connect(DB_PATH)
    match_ids = pd.read_sql_query(f"""
        SELECT match_id FROM matches
        WHERE home_team = '{selected_team}' OR away_team = '{selected_team}';
    """, conn)["match_id"].tolist()
    
    if match_ids:
        match_ids_str = ",".join(map(str, match_ids))
        
        query_players = f"""
            WITH team_events AS (
                SELECT *
                FROM events
                WHERE match_id IN ({match_ids_str})
                AND team = '{selected_team}'
            )
            SELECT 
                player,
                SUM(CASE WHEN type = 'Shot' AND shot_outcome = 'Goal' THEN 1 ELSE 0 END) AS goals,
                SUM(CASE WHEN type = 'Pass' THEN 1 ELSE 0 END) AS passes,
                SUM(CASE WHEN type = 'Pass' AND pass_goal_assist = 1 THEN 1 ELSE 0 END) AS assists,
                SUM(CASE WHEN type = 'Duel' AND duel_outcome = 'Won' THEN 1 ELSE 0 END) AS duels_won,
                ROUND(AVG(CASE WHEN type = 'Shot' THEN shot_statsbomb_xg END), 3) as avg_xg
            FROM team_events
            WHERE player IS NOT NULL
            GROUP BY player
            HAVING goals > 0 OR passes > 50
            ORDER BY goals DESC, passes DESC;
        """
        
        df_players = pd.read_sql_query(query_players, conn)
        conn.close()
        
        if not df_players.empty:
            tab1, tab2 = st.tabs(["📋 Full Stats", "🏆 Top Performers"])
            
            with tab1:
                st.dataframe(df_players, use_container_width=True, hide_index=True)
            
            with tab2:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("#### ⚽ Top Scorers")
                    top_scorers = df_players.nlargest(5, 'goals')[['player', 'goals']]
                    for _, row in top_scorers.iterrows():
                        if row['goals'] > 0:
                            st.write(f"**{row['player']}** - {int(row['goals'])} goals")
                
                with col2:
                    st.markdown("#### 🎯 Top Assist Providers")
                    top_assists = df_players.nlargest(5, 'assists')[['player', 'assists']]
                    for _, row in top_assists.iterrows():
                        if row['assists'] > 0:
                            st.write(f"**{row['player']}** - {int(row['assists'])} assists")
                
                with col3:
                    st.markdown("#### 📦 Most Passes")
                    top_passers = df_players.nlargest(5, 'passes')[['player', 'passes']]
                    for _, row in top_passers.iterrows():
                        st.write(f"**{row['player']}** - {int(row['passes'])} passes")
                
                with col4:
                    st.markdown("#### 🥊 Most Duels Won")
                    top_duels = df_players.nlargest(5, 'duels_won')[['player', 'duels_won']]
                    for _, row in top_duels.iterrows():
                        if row['duels_won'] > 0:
                            st.write(f"**{row['player']}** - {int(row['duels_won'])} won")

# ------------------------------
# 3B. Match Analysis
# ------------------------------
elif selected_team and selected_match and selected_match != "📊 Team Overview":
    
    # Match header with score
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"<h2 style='text-align: right;'>{home_team}</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{home_score} - {away_score}</h1>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<h2 style='text-align: left;'>{away_team}</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Match summary stats
    summary_stats = get_match_summary_stats(match_id, home_team, away_team)
    
    st.markdown("### 📊 Match Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {home_team}")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Passes", f"{summary_stats[home_team]['passes']:.0f}")
            st.metric("Shots", f"{summary_stats[home_team]['shots']:.0f}")
        with metric_col2:
            st.metric("Goals", f"{summary_stats[home_team]['goals']:.0f}")
            st.metric("Avg xG", f"{summary_stats[home_team]['avg_xg']:.3f}" if summary_stats[home_team]['avg_xg'] else "0.000")
        with metric_col3:
            st.metric("Pass Acc.", f"{summary_stats[home_team]['pass_accuracy']:.1f}%" if summary_stats[home_team]['pass_accuracy'] else "0%")
            st.metric("Fouls", f"{summary_stats[home_team]['fouls']:.0f}")
    
    with col2:
        st.markdown(f"#### {away_team}")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Passes", f"{summary_stats[away_team]['passes']:.0f}")
            st.metric("Shots", f"{summary_stats[away_team]['shots']:.0f}")
        with metric_col2:
            st.metric("Goals", f"{summary_stats[away_team]['goals']:.0f}")
            st.metric("Avg xG", f"{summary_stats[away_team]['avg_xg']:.3f}" if summary_stats[away_team]['avg_xg'] else "0.000")
        with metric_col3:
            st.metric("Pass Acc.", f"{summary_stats[away_team]['pass_accuracy']:.1f}%" if summary_stats[away_team]['pass_accuracy'] else "0%")
            st.metric("Fouls", f"{summary_stats[away_team]['fouls']:.0f}")
    
    st.markdown("---")
    
    # Tabbed interface for different analyses
    tab1, tab2, tab3 = st.tabs(["📈 Timeline Analysis", "🎯 Shot Analysis", "🗺️ Position Heatmaps"])
    
    with tab1:
        st.markdown("### Match Timeline - Events per 5-Minute Interval")
        
        col1, col2 = st.columns(2)
        with col1:
            df_pass = get_pass_intervals(match_id)
            create_interval_chart(df_pass, "pass_count", "Passes Over Time", "Number of Passes")
        
        with col2:
            df_shot = get_shot_intervals(match_id)
            create_interval_chart(df_shot, "shot_count", "Shots Over Time", "Number of Shots")
        
        df_receive = get_receive_intervals(match_id)
        create_interval_chart(df_receive, "receive_count", "Ball Receipts Over Time", "Number of Receipts")
    
    with tab2:
        st.markdown("### 🎯 Shot Maps (Hover for Details)")
        df_shots = get_shots_with_xg(match_id)
        
        if not df_shots.empty:
            col1, col2 = st.columns(2)
            with col1:
                display_team_header(home_team)
                plot_interactive_shot_map(df_shots, home_team)
            with col2:
                display_team_header(away_team)
                plot_interactive_shot_map(df_shots, away_team)
        else:
            st.warning("No shot data with xG values found for this match.")
    
    with tab3:
        df_events = get_events(match_id)
        
        if "team" in df_events.columns:
            st.markdown("### Pass Distribution")
            col1, col2 = st.columns(2)
            with col1:
                display_team_header(home_team)
                plot_heatmap(df_events[df_events["team"] == home_team], "Pass", f"{home_team} Passes")
            with col2:
                display_team_header(away_team)
                plot_heatmap(df_events[df_events["team"] == away_team], "Pass", f"{away_team} Passes")
            
            st.markdown("### Ball Receipt Zones")
            col1, col2 = st.columns(2)
            with col1:
                display_team_header(home_team)
                plot_heatmap(df_events[df_events["team"] == home_team], "Ball Receipt", f"{home_team} Receipts")
            with col2:
                display_team_header(away_team)
                plot_heatmap(df_events[df_events["team"] == away_team], "Ball Receipt", f"{away_team} Receipts")