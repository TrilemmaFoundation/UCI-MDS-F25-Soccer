import pandas as pd
import streamlit as st

# File paths (adjust these relative to where your Streamlit app runs)
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELD_TILT_CSV_PATH = ROOT_DIR + "/metrics/PPDA_FieldTilt/field_tilt_per_match.csv"
PPDA_CSV_PATH = ROOT_DIR + "/metrics/PPDA_FieldTilt/ppda_per_match.csv"
FIELD_TILT_AVG_CSV_PATH = ROOT_DIR + "/metrics/PPDA_FieldTilt/field_tilt_team_average.csv"
PPDA_AVG_CSV_PATH = ROOT_DIR + "/metrics/PPDA_FieldTilt/ppda_team_average.csv"


@st.cache_data
def get_field_tilt_data(match_id):
    """Return field tilt data for a specific match"""
    try:
        df_field_tilt = pd.read_csv(FIELD_TILT_CSV_PATH)
        match_data = df_field_tilt[df_field_tilt['match_id'] == match_id]

        if match_data.empty:
            return None

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
    """Return PPDA data for a specific match"""
    try:
        df_ppda = pd.read_csv(PPDA_CSV_PATH)
        match_data = df_ppda[df_ppda['match_id'] == match_id]

        if match_data.empty:
            return None

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
    """Return average field tilt per team"""
    try:
        df_field_tilt = pd.read_csv(FIELD_TILT_AVG_CSV_PATH)
        return df_field_tilt.set_index('team')['field_tilt'].to_dict()
    except Exception as e:
        print(f"Error loading team field tilt averages: {e}")
        return {}


@st.cache_data
def get_team_ppda_averages():
    """Return average PPDA per team"""
    try:
        df_ppda = pd.read_csv(PPDA_AVG_CSV_PATH)
        return df_ppda.set_index('team')['PPDA'].to_dict()
    except Exception as e:
        print(f"Error loading team PPDA averages: {e}")
        return {}
