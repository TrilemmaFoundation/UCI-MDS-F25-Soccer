import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import os


# Default field image path
FIELD_IMAGE_PATH = "../imgs/soccer-field.jpg"


# ---------------------------------
# 1️⃣ Interactive Shot Map
# ---------------------------------
def plot_interactive_shot_map(df_shots, team_name):
    """Plot interactive shot map with xG and outcome details (Plotly)."""
    team_shots = df_shots[df_shots["team"] == team_name].copy()

    if team_shots.empty:
        st.info(f"No shots recorded for {team_name}.")
        return

    # Parse coordinates
    team_shots["x"] = team_shots["location"].apply(
        lambda s: float(s.strip("[]").split(",")[0]) if s and isinstance(s, str) else None
    )
    team_shots["y"] = team_shots["location"].apply(
        lambda s: float(s.strip("[]").split(",")[1]) if s and isinstance(s, str) else None
    )
    team_shots = team_shots.dropna(subset=["x", "y", "shot_statsbomb_xg"])

    if team_shots.empty:
        st.info(f"No valid shot location data for {team_name}.")
        return

    # Hover tooltip
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

    # Background field image
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
                    opacity=0.5,
                    layer="below"
                )
            )
        except Exception as e:
            st.warning(f"Could not load field image: {e}")

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=team_shots["x"],
        y=team_shots["y"],
        mode='markers',
        marker=dict(
            size=12,
            color=team_shots["shot_statsbomb_xg"],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="xG"),
            line=dict(width=1, color='black'),
            opacity=0.8
        ),
        text=team_shots["hover_text"],
        hovertemplate='%{text}<extra></extra>',
        name=team_name
    ))

    # Layout styling
    fig.update_layout(
        title=f"{team_name} - Shot Map (xG and Shot Outcomes)",
        xaxis=dict(range=[0, 120], showgrid=False, visible=False),
        yaxis=dict(range=[0, 80], showgrid=False, visible=False),
        height=400,
        width=600,
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------
# 2️⃣ Interval Line Chart
# ---------------------------------
def create_interval_chart(df, y_col, title, ylabel, color1='#1f77b4', color2='#ff7f0e'):
    """Plot 5-minute interval charts for passes, shots, or other events."""
    if df.empty:
        st.warning(f"No data available for {title}")
        return

    pivot_df = df.pivot(index="interval_start", columns="team", values=y_col).fillna(0)
    fig = go.Figure()

    # Add traces for both teams
    colors = [color1, color2]
    for i, team in enumerate(pivot_df.columns):
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[team],
            mode="lines+markers",
            name=team,
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=8)
        ))

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Interval (minutes)",
        yaxis_title=ylabel,
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
