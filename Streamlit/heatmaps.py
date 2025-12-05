import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os

from db_queries import get_passes_by_interval


# Default image path (adjust if needed)
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELD_IMAGE_PATH = ROOT_DIR + "/imgs/soccer-field.jpg"


# ------------------------------
# 1️⃣ Static Event Heatmap
# ------------------------------
def plot_heatmap(df, event_type, title):
    """Plot static heatmap of events (Pass, Shot, etc.) over soccer field."""
    event_df = df[df["type"].str.contains(event_type, case=False, na=False)].copy()

    if event_df.empty:
        st.warning(f"No {event_type} data available.")
        return

    if "location" not in event_df.columns:
        st.warning(f"No location data for {event_type}.")
        return

    # Parse location coordinates
    event_df["x"] = event_df["location"].apply(
        lambda s: float(s.strip("[]").split(",")[0]) if s and isinstance(s, str) else np.nan
    )
    event_df["y"] = event_df["location"].apply(
        lambda s: float(s.strip("[]").split(",")[1]) if s and isinstance(s, str) else np.nan
    )
    event_df = event_df.dropna(subset=["x", "y"])

    if event_df.empty:
        st.warning(f"No valid location data for {event_type}.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    # Background soccer field image
    if os.path.exists(FIELD_IMAGE_PATH):
        try:
            img = plt.imread(FIELD_IMAGE_PATH)
            ax.imshow(img, extent=[0, 120, 0, 80], alpha=0.5)
        except Exception as e:
            print(f"Error loading field image: {e}")

    # 2D histogram to create heatmap
    heatmap, xedges, yedges = np.histogram2d(
        event_df["x"].values,
        event_df["y"].values,
        bins=50,
        range=[[0, 120], [0, 80]]
    )

    # Overlay heatmap
    if heatmap.max() > 0:
        ax.imshow(
            heatmap.T,
            extent=[0, 120, 0, 80],
            origin="lower",
            cmap="turbo",
            alpha=0.6
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    st.pyplot(fig)
    plt.close()
    


# ------------------------------
# 2️⃣ Animated Pass Heatmap
# ------------------------------
def plot_animated_pass_heatmap(match_id, start_minute, end_minute):
    """Plot animated heatmap of passes for a specific 5-minute interval."""
    df_passes = get_passes_by_interval(match_id, start_minute, end_minute)

    if df_passes.empty:
        st.info(f"No passes recorded in minutes {start_minute}-{end_minute}.")
        return

    # Parse coordinates
    df_passes["x"] = df_passes["location"].apply(
        lambda s: float(s.strip("[]").split(",")[0]) if s and isinstance(s, str) else np.nan
    )
    df_passes["y"] = df_passes["location"].apply(
        lambda s: float(s.strip("[]").split(",")[1]) if s and isinstance(s, str) else np.nan
    )
    df_passes = df_passes.dropna(subset=["x", "y"])

    if df_passes.empty:
        st.info(f"No valid pass coordinates in {start_minute}-{end_minute} min.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Background field
    if os.path.exists(FIELD_IMAGE_PATH):
        try:
            img = plt.imread(FIELD_IMAGE_PATH)
            ax.imshow(img, extent=[0, 120, 0, 80], alpha=0.5)
        except Exception as e:
            print(f"Error loading field image: {e}")

    # Heatmap overlay
    heatmap, xedges, yedges = np.histogram2d(
        df_passes["x"].values,
        df_passes["y"].values,
        bins=50,
        range=[[0, 120], [0, 80]]
    )

    if heatmap.max() > 0:
        im = ax.imshow(
            heatmap.T,
            extent=[0, 120, 0, 80],
            origin="lower",
            cmap="YlOrRd",
            alpha=0.7,
            interpolation="bilinear"
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Number of Passes", rotation=270, labelpad=15)

    # Styling
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Pass Heatmap: {start_minute}-{end_minute} min ({len(df_passes)} passes)",
        fontsize=14, fontweight="bold", pad=20
    )

    # Match time label
    ax.text(
        60, -5,
        f"{start_minute}' - {end_minute}'",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    st.pyplot(fig)
    plt.close()
