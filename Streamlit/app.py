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

# Custom CSS Styling
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
    .definition-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .feature-box {
        background-color: #e8eaf6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# 1. Database Connection
# ------------------------------
DB_PATH = "../statsbomb/statsbomb_euro2020.db"
FLAGS_PATH = "../imgs"
FIELD_IMAGE_PATH = "../imgs/soccer-field.jpg"
FIELD_TILT_CSV_PATH = "../metrics/PPDA_FieldTilt/field_tilt_per_match.csv"
PPDA_CSV_PATH = "../metrics/PPDA_FieldTilt/ppda_per_match.csv"
FIELD_TILT_AVG_CSV_PATH = "../metrics/PPDA_FieldTilt/field_tilt_team_average.csv"
PPDA_AVG_CSV_PATH = "../metrics/PPDA_FieldTilt/ppda_team_average.csv"

# ------------------------------
# Country to Flag Mapping
# ------------------------------
def get_country_flag_code(country_name):
    """Map country names to flag image codes"""
    country_flag_mapping = {
        'Turkey': 'tr', 'Italy': 'it', 'Denmark': 'dk', 'Finland': 'fi',
        'Belgium': 'be', 'Russia': 'ru', 'Wales': 'gb-wls', 'Switzerland': 'ch',
        'England': 'gb-eng', 'Croatia': 'hr', 'Netherlands': 'nl', 'Ukraine': 'ua',
        'Austria': 'at', 'North Macedonia': 'mk', 'Scotland': 'gb-sct',
        'Czech Republic': 'cz', 'Poland': 'pl', 'Slovakia': 'sk', 'Spain': 'es',
        'Sweden': 'se', 'France': 'fr', 'Germany': 'de', 'Hungary': 'hu',
        'Portugal': 'pt'
    }
    return country_flag_mapping.get(country_name, 'unknown')

def get_flag_path(country_name):
    """Get the full path to the flag image for a country"""
    flag_code = get_country_flag_code(country_name)
    flag_path = os.path.join(FLAGS_PATH, f"{flag_code}.png")
    return flag_path

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

@st.cache_data
def get_passes_by_interval(match_id, start_minute, end_minute):
    """Get pass locations for a specific time interval"""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT 
            team,
            location,
            minute,
            player
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
    """Get the maximum minute in the match to determine duration"""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT MAX(minute) as max_minute
        FROM events
        WHERE match_id = {match_id};
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    max_min = df.iloc[0]['max_minute']
    # Round up to nearest 5-minute interval and convert to int
    return int(((max_min // 5) + 1) * 5) if max_min else 90

@st.cache_data
def get_field_tilt_data(match_id):
    """Get field tilt data for a specific match"""
    try:
        df_field_tilt = pd.read_csv(FIELD_TILT_CSV_PATH)
        match_data = df_field_tilt[df_field_tilt['match_id'] == match_id]
        
        if match_data.empty:
            return None
        
        # Create a dictionary with team as key
        field_tilt_dict = {}
        for _, row in match_data.iterrows():
            field_tilt_dict[row['team']] = {
                'final_third_passes': int(row['final_third_passes']),
                'field_tilt': float(row['field_tilt'])
            }
        
        return field_tilt_dict
    except Exception as e:
        print(f"Error loading field tilt data: {e}")
        return None

@st.cache_data
def get_ppda_data(match_id):
    """Get PPDA data for a specific match"""
    try:
        df_ppda = pd.read_csv(PPDA_CSV_PATH)
        match_data = df_ppda[df_ppda['match_id'] == match_id]
        
        if match_data.empty:
            return None
        
        # Create a dictionary with team as key
        ppda_dict = {}
        for _, row in match_data.iterrows():
            ppda_dict[row['team']] = {
                'passes_opponent': int(row['passes_opponent']),
                'def_actions': int(row['def_actions']),
                'ppda': float(row['PPDA'])
            }
        
        return ppda_dict
    except Exception as e:
        print(f"Error loading PPDA data: {e}")
        return None

@st.cache_data
def get_team_field_tilt_averages():
    """Get average field tilt for all teams"""
    try:
        df_field_tilt = pd.read_csv(FIELD_TILT_AVG_CSV_PATH)
        return df_field_tilt.set_index('team')['field_tilt'].to_dict()
    except Exception as e:
        print(f"Error loading team field tilt averages: {e}")
        return {}

@st.cache_data
def get_team_ppda_averages():
    """Get average PPDA for all teams"""
    try:
        df_ppda = pd.read_csv(PPDA_AVG_CSV_PATH)
        return df_ppda.set_index('team')['PPDA'].to_dict()
    except Exception as e:
        print(f"Error loading team PPDA averages: {e}")
        return {}
    
@st.cache_data
def get_team_comparison_data():
    """Get comprehensive team comparison data"""
    conn = sqlite3.connect(DB_PATH)
    
    # Get all teams
    teams_query = """
        SELECT DISTINCT home_team AS team FROM matches
        UNION
        SELECT DISTINCT away_team AS team FROM matches
        ORDER BY team;
    """
    teams = pd.read_sql_query(teams_query, conn)["team"].tolist()
    
    comparison_data = {}
    
    for team in teams:
        # Get match IDs for the team
        match_ids_query = f"""
            SELECT match_id FROM matches
            WHERE home_team = '{team}' OR away_team = '{team}';
        """
        match_ids_df = pd.read_sql_query(match_ids_query, conn)
        match_ids = match_ids_df["match_id"].tolist()
        
        if not match_ids:
            continue
            
        match_ids_str = ",".join(map(str, match_ids))
        
        # Comprehensive team stats
        stats_query = f"""
            WITH team_events AS (
                SELECT *
                FROM events
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
            
            # Calculate derived metrics
            stats['scoring_accuracy'] = (stats['goals_scored'] / stats['total_shots'] * 100) if stats['total_shots'] > 0 else 0
            stats['xg_efficiency'] = (stats['goals_scored'] / stats['total_xg']) if stats['total_xg'] > 0 else 0
            stats['matches_played'] = len(match_ids)
            stats['goals_per_match'] = stats['goals_scored'] / stats['matches_played'] if stats['matches_played'] > 0 else 0
            
            comparison_data[team] = stats
    
    conn.close()
    return comparison_data

@st.cache_data
def get_team_ranking_data(comparison_data, metric, ascending=False, min_matches=1):
    """Get ranked team data for a specific metric"""
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

# ------------------------------
# Helper Functions
# ------------------------------
def display_team_header(team_name):
    """Display team name with flag"""
    col_flag, col_text = st.columns([1, 5])
    flag_path = get_flag_path(team_name)
    with col_flag:
        if os.path.exists(flag_path):
            st.image(flag_path, width=50)
        else:
            st.write("Flag")
    with col_text:
        st.markdown(f"### {team_name}")

def plot_heatmap(df, event_type, title):
    """Matplotlib heatmap overlaid on soccer field image"""
    event_df = df[df["type"].str.contains(event_type, case=False, na=False)].copy()

    if event_df.empty:
        st.warning(f"No {event_type} data available")
        return
    
    if "location" not in event_df.columns:
        st.warning(f"No location data available for {event_type}")
        return

    # Parse location data
    event_df["x"] = event_df["location"].apply(
        lambda s: float(s.strip("[]").split(",")[0]) if pd.notnull(s) and s else np.nan
    )
    event_df["y"] = event_df["location"].apply(
        lambda s: float(s.strip("[]").split(",")[1]) if pd.notnull(s) and s else np.nan
    )
    
    # Drop rows with NaN coordinates
    event_df = event_df.dropna(subset=["x", "y"])
    
    if event_df.empty:
        st.warning(f"No valid location data for {event_type}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    
    if os.path.exists(FIELD_IMAGE_PATH):
        try:
            img = plt.imread(FIELD_IMAGE_PATH)
            ax.imshow(img, extent=[0, 120, 0, 80], alpha=0.5)
        except Exception as e:
            print(f"Error loading field image: {e}")
    
    # Create heatmap
    heatmap, xedges, yedges = np.histogram2d(
        event_df["x"].values, 
        event_df["y"].values, 
        bins=50, 
        range=[[0, 120], [0, 80]]
    )
    
    # Only plot if there's data
    if heatmap.max() > 0:
        ax.imshow(heatmap.T, extent=[0, 120, 0, 80], origin="lower", cmap="turbo", alpha=0.6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold')
    st.pyplot(fig)
    plt.close()

def plot_animated_pass_heatmap(match_id, start_minute, end_minute):
    """Plot pass heatmap for a specific time interval"""
    df_passes = get_passes_by_interval(match_id, start_minute, end_minute)
    
    if df_passes.empty:
        st.info(f"No passes recorded in minutes {start_minute}-{end_minute}")
        return
    
    # Parse location data
    df_passes["x"] = df_passes["location"].apply(
        lambda s: float(s.strip("[]").split(",")[0]) if pd.notnull(s) and s else np.nan
    )
    df_passes["y"] = df_passes["location"].apply(
        lambda s: float(s.strip("[]").split(",")[1]) if pd.notnull(s) and s else np.nan
    )
    
    # Drop rows with NaN coordinates
    df_passes = df_passes.dropna(subset=["x", "y"])
    
    if df_passes.empty:
        st.info(f"No valid pass locations in minutes {start_minute}-{end_minute}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Load and display soccer field image as background
    if os.path.exists(FIELD_IMAGE_PATH):
        try:
            img = plt.imread(FIELD_IMAGE_PATH)
            ax.imshow(img, extent=[0, 120, 0, 80], aspect='auto', alpha=0.5)
        except Exception as e:
            print(f"Error loading field image: {e}")
    
    # Create heatmap
    heatmap, xedges, yedges = np.histogram2d(
        df_passes["x"].values,
        df_passes["y"].values,
        bins=50,
        range=[[0, 120], [0, 80]]
    )
    
    # Only plot if there's data
    if heatmap.max() > 0:
        im = ax.imshow(
            heatmap.T,
            extent=[0, 120, 0, 80],
            origin="lower",
            cmap="YlOrRd",
            alpha=0.7,
            interpolation='bilinear'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Number of Passes', rotation=270, labelpad=15)
    
    # Styling
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Pass Heatmap: Minutes {start_minute}-{end_minute} ({len(df_passes)} passes)",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add match time indicator
    ax.text(
        60, -5,
        f"Match Time: {start_minute}' - {end_minute}'",
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
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
                    y=77,
                    sizex=120,
                    sizey=77,
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
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        height=400,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0)
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
    st.markdown("# Soccer Analytics")
    st.markdown("### UEFA Euro 2020")
    st.markdown("---")
    
    # Navigation
    page_options = ["Home", "Team Analysis", "Match Analysis"]
    selected_page = st.radio("Navigation", page_options, label_visibility="collapsed")
    
    st.markdown("---")
    
    if selected_page != "Home":
        selected_team = st.selectbox("Select a Team", get_teams(), key="team_select")
        
        if selected_page == "Match Analysis" and selected_team:
            matches = get_matches(selected_team)
            match_options = [
                f"{row['home_team']} vs {row['away_team']}"
                for _, row in matches.iterrows()
            ]
            
            selected_match = st.selectbox("Select a Match", match_options, key="match_select")
            
            if selected_match:
                match_row = matches[matches.apply(
                    lambda row: f"{row['home_team']} vs {row['away_team']}" == selected_match, axis=1
                )].iloc[0]
                match_id = match_row["match_id"]
                home_team = match_row["home_team"]
                away_team = match_row["away_team"]
                home_score = match_row["home_score"]
                away_score = match_row["away_score"]
                match_date = match_row["match_date"]
                stage = match_row["competition_stage"]
                
                st.markdown("---")
                st.markdown("### Match Info")
                
                # Display flags in sidebar
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    home_flag_path = get_flag_path(home_team)
                    if os.path.exists(home_flag_path):
                        st.image(home_flag_path, width=40)
                with col2:
                    st.markdown("**VS**")
                with col3:
                    away_flag_path = get_flag_path(away_team)
                    if os.path.exists(away_flag_path):
                        st.image(away_flag_path, width=40)
                
                st.write(f"**Date:** {match_date}")
                st.write(f"**Stage:** {stage}")
                st.write(f"**Score:** {home_score} - {away_score}")

# ------------------------------
# 3. Main Views
# ------------------------------

# ------------------------------
# 3A. Home Page
# ------------------------------
if selected_page == "Home":
    st.markdown('<h1 class="main-header">Soccer Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### UEFA Euro 2020 - Advanced Match & Team Analysis")
    
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    Welcome to the **Soccer Analytics Dashboard**, a comprehensive platform for analyzing UEFA Euro 2020 matches 
    using advanced metrics and visualizations. This dashboard leverages StatsBomb's open-source event data to provide 
    deep insights into team and player performance.
    """)
    
    st.markdown("---")
    
    # Key Metrics Definitions
    st.markdown("## Key Metrics Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="definition-box">
        <h3>Expected Goals (xG)</h3>
        <p><strong>Definition:</strong> A statistical measure that quantifies the quality of a scoring chance based on historical data.</p>
        <p><strong>Range:</strong> 0.0 to 1.0, where 1.0 represents a certain goal</p>
        <p><strong>Factors:</strong> Distance from goal, angle, body part used, assist type, defensive pressure</p>
        <p><strong>Use Case:</strong> Evaluates shot quality and team attacking efficiency independent of actual outcomes</p>
        </div>
        """, unsafe_allow_html=True)
        
        # st.markdown("""
        # <div class="definition-box">
        # <h3>Pass Accuracy</h3>
        # <p><strong>Definition:</strong> Percentage of successful passes completed by a team</p>
        # <p><strong>Calculation:</strong> (Successful Passes / Total Passes) × 100</p>
        # <p><strong>Significance:</strong> Indicates ball retention and possession quality</p>
        # </div>
        # """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="definition-box">
        <h3>PPDA (Passes Per Defensive Action)</h3>
        <p><strong>Definition:</strong> Measures how many opponent passes a team allows before applying defensive pressure</p>
        <p><strong>Calculation:</strong> Opponent Passes / (Pressures + Tackles + Interceptions + Fouls)</p>
        <p><strong>Interpretation:</strong> Lower PPDA indicates more aggressive pressing and higher defensive intensity</p>
        <p><strong>Use Case:</strong> Evaluates pressing effectiveness and defensive proactivity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="definition-box">
        <h3>Expected Threat (xT)</h3>
        <p><strong>Definition:</strong> Measures the value of ball progression by quantifying how actions move the ball from low-value to high-value zones</p>
        <p><strong>Application:</strong> Evaluates progressive passing and ball-carrying effectiveness</p>
        <p><strong>Innovation:</strong> Goes beyond simple pass completion to assess territorial advancement</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="definition-box">
        <h3>Field Tilt</h3>
        <p><strong>Definition:</strong> Measures territorial dominance by calculating the percentage of final third passes made by a team</p>
        <p><strong>Calculation:</strong> (Team's Final Third Passes / Total Final Third Passes) × 100</p>
        <p><strong>Significance:</strong> Indicates which team is controlling the game in attacking areas and applying sustained pressure</p>
        <p><strong>Interpretation:</strong> Higher field tilt percentage suggests greater attacking territorial control</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dashboard Features
    st.markdown("## Dashboard Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h4>Team Analysis</h4>
        <ul>
        <li>Tournament-wide statistics</li>
        <li>Player performance rankings</li>
        <li>Goals, assists, and duels won</li>
        <li>Aggregate xG metrics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h4>Match Analysis</h4>
        <ul>
        <li>Interactive shot maps with xG</li>
        <li>Timeline event tracking</li>
        <li>Pass and receipt heatmaps</li>
        <li>Head-to-head comparisons</li>
        <li>Animated pass heatmaps</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
        <h4>Visualizations</h4>
        <ul>
        <li>Real-time interactive charts</li>
        <li>5-minute interval tracking</li>
        <li>Spatial event distributions</li>
        <li>Hoverable data points</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("## Getting Started")
    st.markdown("""
    1. **Navigate** using the sidebar to choose between Team Analysis or Match Analysis
    2. **Select a team** from the dropdown to explore their tournament performance
    3. **Choose a match** (in Match Analysis mode) to dive into specific game details
    4. **Interact** with visualizations by hovering over charts and heatmaps for detailed information
    """)
    
    st.markdown("---")
    
    # Data Source
    st.markdown("## Data Source")
    st.markdown("""
    This dashboard uses **StatsBomb Open Data** for UEFA Euro 2020, which includes:
    - 51 tournament matches
    - 192,692+ event records
    - 2.5+ million tracking frames
    - Comprehensive xG calculations for all shots
    
    **License:** CC BY-NC 4.0 (Non-Commercial Use)
    """)

# ------------------------------
# 3B. Team Analysis
# ------------------------------
elif selected_page == "Team Analysis" and selected_team:
    st.markdown('<h1 class="main-header">Team Analysis</h1>', unsafe_allow_html=True)
    
    # Load all team comparison data
    comparison_data = get_team_comparison_data()
    field_tilt_averages = get_team_field_tilt_averages()
    ppda_averages = get_team_ppda_averages()
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Team Overview", "Team Comparisons", "Tournament Rankings"])
    
    with tab1:
        # Individual Team Overview (existing functionality)
        st.markdown(f"## {selected_team} - Tournament Statistics")
        
        if selected_team in comparison_data:
            team_stats = comparison_data[selected_team]
            
            st.markdown("### Team Performance Metrics")
            
            # Create 7 columns for the metrics grid
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            
            with col1:
                st.metric("Matches Played", f"{team_stats['matches_played']}")
                st.metric("Total Passes", f"{team_stats['total_passes']:,}")
            with col2:
                st.metric("Total Shots", f"{team_stats['total_shots']}")
                st.metric("Goals Scored", f"{team_stats['goals_scored']}")
            with col3:
                st.metric("Total xG", f"{team_stats['total_xg']:.3f}")
                st.metric("Goals/Match", f"{team_stats['goals_per_match']:.2f}")
            with col4:
                st.metric("Scoring Accuracy", f"{team_stats['scoring_accuracy']:.1f}%")
                st.metric("xG Efficiency", f"{team_stats['xg_efficiency']:.2f}")
            with col5:
                st.metric("Pass Accuracy", f"{team_stats['pass_accuracy']:.1f}%")
                st.metric("Assists", f"{team_stats['assists']}")
            with col6:
                # Average Field Tilt
                avg_field_tilt = field_tilt_averages.get(selected_team)
                if avg_field_tilt is not None:
                    st.metric("Avg Field Tilt", f"{avg_field_tilt:.1f}%")
                else:
                    st.metric("Avg Field Tilt", "N/A")
                st.metric("Duels Won", f"{team_stats['duels_won']}")
            with col7:
                # Average PPDA
                avg_ppda = ppda_averages.get(selected_team)
                if avg_ppda is not None:
                    st.metric("Avg PPDA", f"{avg_ppda:.2f}")
                else:
                    st.metric("Avg PPDA", "N/A")
                st.metric("Interceptions", f"{team_stats['interceptions']}")
            
            # Add explanation for the new metrics
            with st.expander("ℹ️ About Advanced Metrics"):
                st.markdown("""
                **Scoring Accuracy**: Percentage of shots that result in goals (Goals / Total Shots × 100)
                
                **xG Efficiency**: Goals scored relative to expected goals (Goals / Total xG). Values > 1 indicate overperformance.
                
                **Field Tilt**: Percentage of final third passes made by a team. Higher values indicate greater territorial dominance.
                
                **PPDA (Passes Per Defensive Action)**: Measures pressing intensity. Lower values indicate more aggressive defensive pressure.
                """)
            
            st.markdown("---")
        
        # Player-level stats (existing code)
        st.markdown("### Player Performance")
        
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
                    SUM(CASE WHEN type = 'Pass' AND pass_assisted_shot_id IS NOT NULL THEN 1 ELSE 0 END) AS assists,
                    SUM(CASE WHEN type = 'Duel' AND duel_outcome = 'Won' THEN 1 ELSE 0 END) AS duels_won,
                    ROUND(SUM(CASE WHEN type = 'Shot' THEN shot_statsbomb_xg END), 3) as total_xg
                FROM team_events
                WHERE player IS NOT NULL
                GROUP BY player
                HAVING goals > 0 OR passes > 50
                ORDER BY goals DESC, passes DESC;
            """
            
            df_players = pd.read_sql_query(query_players, conn)
            conn.close()
            
            if not df_players.empty:
                player_tab1, player_tab2 = st.tabs(["Full Statistics", "Top Performers"])
                
                with player_tab1:
                    st.dataframe(df_players, use_container_width=True, hide_index=True)
                
                with player_tab2:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("#### Top Scorers")
                        top_scorers = df_players.nlargest(5, 'goals')[['player', 'goals']]
                        for _, row in top_scorers.iterrows():
                            if row['goals'] > 0:
                                st.write(f"**{row['player']}** - {int(row['goals'])} goals")
                    
                    with col2:
                        st.markdown("#### Top Assist Providers")
                        top_assists = df_players.nlargest(5, 'assists')[['player', 'assists']]
                        if not top_assists.empty and top_assists['assists'].sum() > 0:
                            for _, row in top_assists.iterrows():
                                if row['assists'] > 0:
                                    st.write(f"**{row['player']}** - {int(row['assists'])} assists")
                        else:
                            st.write("No assist data available")
                    
                    with col3:
                        st.markdown("#### Most Passes")
                        top_passers = df_players.nlargest(5, 'passes')[['player', 'passes']]
                        for _, row in top_passers.iterrows():
                            st.write(f"**{row['player']}** - {int(row['passes'])} passes")
                    
                    with col4:
                        st.markdown("#### Most Duels Won")
                        top_duels = df_players.nlargest(5, 'duels_won')[['player', 'duels_won']]
                        if not top_duels.empty and top_duels['duels_won'].sum() > 0:
                            for _, row in top_duels.iterrows():
                                if row['duels_won'] > 0:
                                    st.write(f"**{row['player']}** - {int(row['duels_won'])} won")
                        else:
                            st.write("No duel data available")
    
    with tab2:
        # Team Comparisons
        st.markdown("## Team Performance Comparisons")
        
        if selected_team in comparison_data:
            # Comparison metrics selection
            col1, col2 = st.columns(2)
            
            with col1:
                comparison_metric = st.selectbox(
                    "Select Metric to Compare",
                    [
                        "scoring_accuracy", "goals_scored", "goals_per_match",
                        "total_xg", "xg_efficiency", "pass_accuracy",
                        "total_passes", "duels_won", "interceptions"
                    ],
                    format_func=lambda x: {
                        "scoring_accuracy": "Scoring Accuracy (%)",
                        "goals_scored": "Total Goals",
                        "goals_per_match": "Goals per Match",
                        "total_xg": "Total Expected Goals (xG)",
                        "xg_efficiency": "xG Efficiency",
                        "pass_accuracy": "Pass Accuracy (%)",
                        "total_passes": "Total Passes",
                        "duels_won": "Duels Won",
                        "interceptions": "Interceptions"
                    }[x]
                )
            
            with col2:
                show_top_n = st.slider("Number of Teams to Show", 5, 20, 10)
            
            # Get ranking data
            ascending_metrics = ["ppda"]  # Metrics where lower values are better
            ascending_order = comparison_metric in ascending_metrics

            ranked_df = get_team_ranking_data(comparison_data, comparison_metric, ascending_order)

            # Reset index for proper ranking and add rank column
            ranked_df = ranked_df.reset_index(drop=True)
            ranked_df['rank'] = ranked_df.index + 1
            
            if not ranked_df.empty:
                # Filter to top N teams for the chart
                top_teams = ranked_df.head(show_top_n)
                
                # Create bar chart
                fig = go.Figure()
                
                # Highlight selected team
                colors = []
                for team in top_teams['team']:
                    if team == selected_team:
                        colors.append('#1f77b4')  # Highlight color
                    else:
                        colors.append('#ff7f0e')  # Default color
                
                fig.add_trace(go.Bar(
                    x=top_teams['team'],
                    y=top_teams['value'],
                    marker_color=colors,
                    hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<br>Matches: %{customdata}',
                    customdata=top_teams['matches_played']
                ))
                
                metric_names = {
                    "scoring_accuracy": "Scoring Accuracy (%)",
                    "goals_scored": "Total Goals",
                    "goals_per_match": "Goals per Match",
                    "total_xg": "Total Expected Goals (xG)",
                    "xg_efficiency": "xG Efficiency",
                    "pass_accuracy": "Pass Accuracy (%)",
                    "total_passes": "Total Passes",
                    "duels_won": "Duels Won",
                    "interceptions": "Interceptions"
                }
                
                fig.update_layout(
                    title=f"Top {show_top_n} Teams - {metric_names[comparison_metric]}",
                    xaxis_title="Team",
                    yaxis_title=metric_names[comparison_metric],
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Show selected team's position
                selected_team_info = ranked_df[ranked_df['team'] == selected_team]
                if not selected_team_info.empty:
                    rank_position = selected_team_info['rank'].iloc[0]
                    team_value = selected_team_info['value'].iloc[0]
                    total_teams = len(ranked_df)
                    
                    # Format value based on metric type
                    if "accuracy" in comparison_metric:
                        formatted_value = f"{team_value:.1f}%"
                    elif "xg" in comparison_metric or "efficiency" in comparison_metric:
                        formatted_value = f"{team_value:.3f}"
                    else:
                        formatted_value = f"{team_value:.0f}"
                    
                    st.info(
                        f"**{selected_team}** is ranked **#{rank_position}** out of **{total_teams}** teams "
                        f"in {metric_names[comparison_metric]} with a value of **{formatted_value}**"
                    )
                
    with tab3:
        # Tournament Rankings
        st.markdown("## Tournament Leaderboards")
        
        # Define ranking categories
        ranking_categories = {
            "attack": ["goals_scored", "goals_per_match", "scoring_accuracy", "total_xg", "xg_efficiency"],
            "possession": ["total_passes", "pass_accuracy"],
            "defense": ["duels_won", "interceptions"]
        }
        
        category = st.selectbox(
            "Select Category",
            ["attack", "possession", "defense"],
            format_func=lambda x: {
                "attack": "⚽ Attacking Metrics",
                "possession": "🎯 Possession Metrics", 
                "defense": "🛡️ Defensive Metrics"
            }[x]
        )
        
        st.markdown(f"### Top 5 Teams - {category.title()} Metrics")
        
        for metric in ranking_categories[category]:
            ranked_df = get_team_ranking_data(comparison_data, metric, metric in ["ppda"])
            
            if not ranked_df.empty:
                top_5 = ranked_df.head(5)
                
                metric_names = {
                    "goals_scored": "Total Goals",
                    "goals_per_match": "Goals per Match",
                    "scoring_accuracy": "Scoring Accuracy",
                    "total_xg": "Total xG",
                    "xg_efficiency": "xG Efficiency",
                    "total_passes": "Total Passes",
                    "pass_accuracy": "Pass Accuracy",
                    "duels_won": "Duels Won",
                    "interceptions": "Interceptions"
                }
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown(f"{metric_names[metric]}")
                    for i, (_, row) in enumerate(top_5.iterrows()):
                        medal = ["🥇", "🥈", "🥉", "4.", "5."][i]

                        # Format the value based on the metric type
                        if "accuracy" in metric:
                            formatted_value = f"{row['value']:.1f}%"
                        elif "xg" in metric or "efficiency" in metric:
                            formatted_value = f"{row['value']:.3f}"
                        else:
                            formatted_value = f"{row['value']:.0f}"

                        st.write(f"{medal} {row['team']} - {formatted_value}")
                
                with col2:
                    # Show selected team's position if in top 20
                    selected_team_rank = ranked_df[ranked_df['team'] == selected_team].index
                    if not selected_team_rank.empty:
                        rank_position = selected_team_rank[0] + 1
                        if rank_position <= 20:
                            team_value = ranked_df[ranked_df['team'] == selected_team]['value'].iloc[0]
                            
                            # Format the value based on the metric type
                            if "accuracy" in metric:
                                formatted_team_value = f"{team_value:.1f}%"
                            elif "xg" in metric or "efficiency" in metric:
                                formatted_team_value = f"{team_value:.3f}"
                            else:
                                formatted_team_value = f"{team_value:.0f}"
                            
                            st.metric(
                                f"{selected_team} Rank", 
                                f"#{rank_position}",
                                f"Value: {formatted_team_value}"
            )
                
                st.markdown("---")

# ------------------------------
# 3C. Match Analysis
# ------------------------------
elif selected_page == "Match Analysis" and selected_team and selected_match:
    st.markdown('<h1 class="main-header">Match Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Match header with score and flags
    col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
    
    with col1:
        flag_path = get_flag_path(home_team)
        if os.path.exists(flag_path):
            st.image(flag_path, width=80)
    
    with col2:
        st.markdown(f"<h2 style='text-align: right; margin: 0;'>{home_team}</h2>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"<h1 style='text-align: center; color: #1f77b4; margin: 0;'>{home_score} - {away_score}</h1>", unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"<h2 style='text-align: left; margin: 0;'>{away_team}</h2>", unsafe_allow_html=True)
    
    with col5:
        flag_path = get_flag_path(away_team)
        if os.path.exists(flag_path):
            st.image(flag_path, width=80)
    
    st.markdown("---")
    
    # Match summary stats
    summary_stats = get_match_summary_stats(match_id, home_team, away_team)
    
    # Get field tilt data
    field_tilt_data = get_field_tilt_data(match_id)
    
    # Get PPDA data
    ppda_data = get_ppda_data(match_id)
    
    st.markdown("### Match Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        display_team_header(home_team)
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Passes", f"{summary_stats[home_team]['passes']:.0f}")
            st.metric("Shots", f"{summary_stats[home_team]['shots']:.0f}")
            st.metric("Goals", f"{summary_stats[home_team]['goals']:.0f}")
        with metric_col2:
            st.metric("Total xG", f"{summary_stats[home_team]['total_xg']:.3f}" if summary_stats[home_team]['total_xg'] else "0.000")
            st.metric("Pass Acc.", f"{summary_stats[home_team]['pass_accuracy']:.1f}%" if summary_stats[home_team]['pass_accuracy'] else "0%")
            st.metric("Fouls", f"{summary_stats[home_team]['fouls']:.0f}")
        with metric_col3:
            if field_tilt_data and home_team in field_tilt_data:
                st.metric("Field Tilt", f"{field_tilt_data[home_team]['field_tilt']:.1f}%")
                st.metric("Final 3rd Passes", f"{field_tilt_data[home_team]['final_third_passes']}")
            else:
                st.metric("Field Tilt", "N/A")
                st.metric("Final 3rd Passes", "N/A")
            
            if ppda_data and home_team in ppda_data:
                st.metric("PPDA", f"{ppda_data[home_team]['ppda']:.2f}")
            else:
                st.metric("PPDA", "N/A")
    
    with col2:
        display_team_header(away_team)
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Passes", f"{summary_stats[away_team]['passes']:.0f}")
            st.metric("Shots", f"{summary_stats[away_team]['shots']:.0f}")
            st.metric("Goals", f"{summary_stats[away_team]['goals']:.0f}")
        with metric_col2:
            st.metric("Total xG", f"{summary_stats[away_team]['total_xg']:.3f}" if summary_stats[away_team]['total_xg'] else "0.000")
            st.metric("Pass Acc.", f"{summary_stats[away_team]['pass_accuracy']:.1f}%" if summary_stats[away_team]['pass_accuracy'] else "0%")
            st.metric("Fouls", f"{summary_stats[away_team]['fouls']:.0f}")
        with metric_col3:
            if field_tilt_data and away_team in field_tilt_data:
                st.metric("Field Tilt", f"{field_tilt_data[away_team]['field_tilt']:.1f}%")
                st.metric("Final 3rd Passes", f"{field_tilt_data[away_team]['final_third_passes']}")
            else:
                st.metric("Field Tilt", "N/A")
                st.metric("Final 3rd Passes", "N/A")
            
            if ppda_data and away_team in ppda_data:
                st.metric("PPDA", f"{ppda_data[away_team]['ppda']:.2f}")
            else:
                st.metric("PPDA", "N/A")

    st.markdown("---")
    
    # Tabbed interface for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Timeline Analysis", "Shot Analysis", "Position Heatmaps", "Animated Pass Map"])
    
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
        st.markdown("### Shot Maps (Hover for Details)")
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
    
    with tab4:
        st.markdown("### Animated Pass Heatmap")
        st.markdown("Use the slider below to explore pass distribution throughout the match in 5-minute intervals.")
        
        # Get match duration
        match_duration = get_match_duration(match_id)
        
        # Initialize session state for slider if not exists
        if 'time_interval' not in st.session_state:
            st.session_state.time_interval = 0
        
        # Create columns for controls
        col1, col2, col3 = st.columns([1, 6, 1])
        
        with col1:
            if st.button("⏮️ Reset", use_container_width=True):
                st.session_state.time_interval = 0
                st.rerun()
        
        with col2:
            # Time interval slider
            time_interval = st.slider(
                "Select Time Interval (minutes)",
                min_value=0,
                max_value=int(match_duration - 5),
                value=int(st.session_state.time_interval),
                step=5,
                format="%d min",
                key="time_slider"
            )
            st.session_state.time_interval = time_interval
        
        with col3:
            if st.button("⏭️ End", use_container_width=True):
                st.session_state.time_interval = match_duration - 5
                st.rerun()
        
        # Display current interval info
        start_min = time_interval
        end_min = time_interval + 5
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Current Interval", f"{start_min}-{end_min} min")
        with col_info2:
            st.metric("Match Duration", f"{match_duration} min")
        with col_info3:
            progress = ((time_interval + 5) / match_duration) * 100
            st.metric("Progress", f"{progress:.0f}%")
        
        st.markdown("---")
        
        # Plot the heatmap for selected interval
        plot_animated_pass_heatmap(match_id, start_min, end_min)
        
        # Add navigation buttons at bottom
        st.markdown("---")
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            if st.button("⏪ Previous Interval", use_container_width=True, disabled=(time_interval == 0)):
                st.session_state.time_interval = max(0, time_interval - 5)
                st.rerun()
        
        with col_btn2:
            if st.button("⏸️ First Half", use_container_width=True):
                st.session_state.time_interval = 0
                st.rerun()
        
        with col_btn3:
            if st.button("⏯️ Second Half", use_container_width=True):
                st.session_state.time_interval = 45
                st.rerun()
        
        with col_btn4:
            if st.button("⏩ Next Interval", use_container_width=True, disabled=(time_interval >= match_duration - 5)):
                st.session_state.time_interval = min(match_duration - 5, time_interval + 5)
                st.rerun()
        
        # Add legend/info box
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            **Controls:**
            - 🎚️ **Slider**: Drag to jump to any 5-minute interval
            - ⏮️ **Reset**: Jump to start of match (0-5 min)
            - ⏭️ **End**: Jump to final interval
            - ⏪ **Previous**: Go back one interval (5 minutes)
            - ⏩ **Next**: Go forward one interval (5 minutes)
            - ⏸️ **First Half**: Jump to start of first half (0-5 min)
            - ⏯️ **Second Half**: Jump to start of second half (45-50 min)
            
            **Understanding the Heatmap:**
            - 🔴 **Red/Orange areas**: High concentration of passes
            - 🟡 **Yellow areas**: Moderate pass activity
            - ⚪ **White areas**: Low or no pass activity
            - The heatmap shows **combined passes from both teams**
            - Each interval displays **5 minutes** of match action
            """)

