import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1. Database Connection
# ------------------------------
DB_PATH = r"C:\Users\Bryan\CodeProjects\D296P-297P\UCI-MDS-F25-Soccer\statsbomb_euro2020.db"
#DB_PATH = "/Users/indrajeet/Documents/Compsci 296p/UCI-MDS-F25-Soccer/statsbomb_euro2020.db"

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
        SELECT match_id, home_team, away_team, match_date
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
    selected_match = st.sidebar.selectbox("Select a Match", match_options)
    match_id = matches.iloc[match_options.index(selected_match)]["match_id"]

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Selected Team:** {selected_team}")
    st.sidebar.write(f"**Match ID:** {match_id}")


# ------------------------------
# 3. Main View
# ------------------------------
st.title("📊 Match Analytics Dashboard")

if selected_team and selected_match:
    st.subheader(f"{selected_match}")

    df_events = get_events(match_id)
    st.write("### Sample of Match Events")
    st.dataframe(df_events.head())

    # ------------------------------
    # Helper: Draw Heatmap
    # ------------------------------
    def plot_heatmap(df, event_type, title):
        # Filter by event type
        event_df = df[df["type"].str.contains(event_type, case=False, na=False)]

        # Extract coordinates (assuming StatsBomb's 120x80 field)
        if "x" in event_df.columns and "y" in event_df.columns:
            x = event_df["x"].astype(float)
            y = event_df["y"].astype(float)
        elif "location" in event_df.columns:
            # Sometimes coordinates stored as text or JSON-like "[x, y]"
            event_df["x"] = event_df["location"].apply(lambda s: float(s.strip("[]").split(",")[0]) if pd.notnull(s) else np.nan)
            event_df["y"] = event_df["location"].apply(lambda s: float(s.strip("[]").split(",")[1]) if pd.notnull(s) else np.nan)
            x, y = event_df["x"], event_df["y"]
        else:
            # Fallback random points
            x = np.random.uniform(0, 120, 100)
            y = np.random.uniform(0, 80, 100)

        # Background field image (optional)
        try:
            img = plt.imread(r"C:\Users\Bryan\CodeProjects\D296P-297P\UCI-MDS-F25-Soccer\imgs\soccer-field.jpg")
            #img = plt.imread("/Users/indrajeet/Documents/Compsci 296p/UCI-MDS-F25-Soccer/imgs")
            has_field_image = True
        except FileNotFoundError:
            img = None
            has_field_image = False

        # Compute heatmap
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[0, 120], [0, 80]])

        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        if has_field_image:
            ax.imshow(img, extent=[-4, 125, -2, 81], aspect='auto')
        im = ax.imshow(heatmap.T, extent=[0, 120, 0, 80],
                       origin='lower', cmap='turbo', alpha=0.5)
        ax.set_title(title)
        st.pyplot(fig)

    # ------------------------------
    # 4. Generate Heatmaps
    # ------------------------------
    st.write("## 🎯 Event Heatmaps")

    plot_heatmap(df_events, "Shot", f"{selected_team} — Shot Heatmap")
    plot_heatmap(df_events, "Pass", f"{selected_team} — Pass Heatmap")
    plot_heatmap(df_events, "Ball Receipt", f"{selected_team} — Receive Heatmap")

    # ------------------------------
    # Add scoreboard to 5 min intervals to get a better analysis
    # Map imgs with country flags
    # Ask about xG numbers
    # Add simple statistics