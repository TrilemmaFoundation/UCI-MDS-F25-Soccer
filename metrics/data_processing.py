"""
Generates training data from statsbomb open repo and uefa sqlite3 database.

Place data files in following structure:

current working directory
├── data/
|   ├── statsbomb_euro2020.db
│   └── statsbomb/
│       └─── data/
└── output/                     # output data will be placed into this folder
"""

import json
import math
import os
import sqlite3

import numpy as np
import pandas as pd

STATSBOMB_DIR = "data/statsbomb"
GOAL_X, GOAL_Y = 120, 40
GOAL_LEFT_Y, GOAL_RIGHT_Y = 36, 44  # 8 yards apart
goal_left = np.array([120, 36])
goal_right = np.array([120, 44])
c = np.hypot(goal_left[0] - goal_right[0], goal_left[1] - goal_right[1])  # constant


def compute_distance(x, y):
    return math.hypot(GOAL_X - x, GOAL_Y - y)


def compute_angle(x, y):
    goal_left, goal_right = (120, 36), (120, 44)
    a = math.hypot(goal_left[0] - x, goal_left[1] - y)
    b = math.hypot(goal_right[0] - x, goal_right[1] - y)
    c = math.hypot(goal_left[0] - goal_right[0], goal_left[1] - goal_right[1])
    try:
        angle = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
    except ValueError:
        angle = 0
    return angle


def compute_distance_vec(x, y):
    return np.hypot(GOAL_X - x, GOAL_Y - y)


def compute_angle_vec(x, y):
    a = np.hypot(goal_left[0] - x, goal_left[1] - y)
    b = np.hypot(goal_right[0] - x, goal_right[1] - y)

    cos_theta = (a * a + b * b - c * c) / (2 * a * b)
    return np.arccos(np.clip(cos_theta, -1, 1))


def count_players(freeze):
    if not freeze:  # handles empty list, None, "", etc.
        return (0, 0)

    num_teammates = sum(1 for p in freeze if p.get("teammate"))
    num_opponents = len(freeze) - num_teammates
    return (num_teammates, num_opponents)


def safe_json_list(s):
    if not s:  # catches "", None, NaN
        return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return []


def extract_shots(match_file: str):
    with open(match_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    shots = []
    for ev in data:
        if ev.get("type", {}).get("name") != "Shot":
            continue
        x, y = ev["location"]
        dist = compute_distance(x, y)
        angle = compute_angle(x, y)
        shot = ev["shot"]
        num_teammates, num_opponents = count_players(shot.get("freeze_frame", []))

        shots.append(
            {
                "match_id": match_file,
                "team": ev["team"]["name"],
                "player": ev["player"]["name"],
                "x": x,
                "y": y,
                "distance": dist,
                "angle": angle,
                "shot_body_part": shot["body_part"]["name"],
                "shot_technique": shot["technique"]["name"],
                "shot_type": shot["type"]["name"],
                "play_pattern": ev["play_pattern"]["name"],
                "num_teammates": num_teammates,
                "num_opponents": num_opponents,
                "goal": 1 if shot["outcome"]["name"] == "Goal" else 0,
                "xG_statsbomb": shot.get("statsbomb_xg", None),
            }
        )
    return shots


def extract_shots_from_db(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = """
    SELECT
        match_id,
        team,
        player,
        location,
        play_pattern,
        shot_end_location,
        shot_statsbomb_xg,
        shot_outcome,
        shot_technique,
        shot_body_part,
        shot_type,
        shot_freeze_frame
    FROM events
    WHERE type = 'Shot'
    """
    rows = cur.execute(query).fetchall()
    cols = [desc[0] for desc in cur.description]

    shots = []
    for row in rows:
        ev = dict(zip(cols, row))
        x, y = json.loads(ev["location"])
        dist = compute_distance(x, y)
        angle = compute_angle(x, y)
        num_teammates, num_opponents = count_players(
            safe_json_list(ev["shot_freeze_frame"])
        )
        goal_flag = 1 if ev.get("shot_outcome") == "Goal" else 0

        shots.append(
            {
                "match_id": ev["match_id"],
                "team": ev["team"],
                "player": ev["player"],
                "x": x,
                "y": y,
                "distance": dist,
                "angle": angle,
                "shot_body_part": ev["shot_body_part"],
                "shot_technique": ev["shot_technique"],
                "shot_type": ev["shot_type"],
                "play_pattern": ev["play_pattern"],
                "num_teammates": num_teammates,
                "num_opponents": num_opponents,
                "goal": goal_flag,
                "xG_statsbomb": ev["shot_statsbomb_xg"],
            }
        )

    conn.close()
    return pd.DataFrame(shots)


def statsbomb_load_league(league_json_file: str, out_csv: str = "", out_dir: str = ""):
    with open(
        os.path.join(STATSBOMB_DIR, "data/matches/", league_json_file),
        "r",
        encoding="utf-8",
    ) as f:
        matches = json.load(f)

    match_ids = [match["match_id"] for match in matches]
    events_filepath = [
        os.path.join(os.path.join(STATSBOMB_DIR, "data/events"), f"{match_id}.json")
        for match_id in match_ids
    ]

    all_features = []
    for events in events_filepath:
        all_features.extend(extract_shots(events))

    df = pd.DataFrame(all_features)
    if out_csv:
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, out_csv), index=False, encoding="utf-8-sig")

    return df


def statsbomb_load_league_xt_full(
    league_json_file: str, out_jsonl: str = "", out_dir: str = ""
):
    with open(
        os.path.join(STATSBOMB_DIR, "data/matches/", league_json_file),
        "r",
        encoding="utf-8",
    ) as f:
        matches = json.load(f)

    match_ids = [match["match_id"] for match in matches]
    events_filepath = [
        os.path.join(os.path.join(STATSBOMB_DIR, "data/events"), f"{match_id}.json")
        for match_id in match_ids
    ]

    all_events = []
    for events in events_filepath:
        with open(events, "r") as f:
            for event in json.load(f):
                if event["type"]["name"] in ["Pass", "Carry", "Shot"]:
                    all_events.append(event)

    if out_jsonl:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, out_jsonl), "w") as f:
            for event in all_events:
                f.write(json.dumps(event) + "\n")

    return all_events


def statsbomb_load_league_xt(
    league_json_file: str, out_jsonl: str = "", out_dir: str = ""
):
    matches_path = os.path.join(STATSBOMB_DIR, "data/matches", league_json_file)
    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    match_ids = [m["match_id"] for m in matches]
    event_files = [
        os.path.join(STATSBOMB_DIR, "data/events", f"{mid}.json") for mid in match_ids
    ]

    all_events = []
    for ef in event_files:
        with open(ef, "r", encoding="utf-8") as f:
            events = json.load(f)
        for e in events:
            t = e["type"]["name"]
            if t not in ("Pass", "Carry", "Shot"):
                continue

            small = {
                "type": t,
                "location": e.get("location"),
            }
            if t == "Pass":
                small["end_location"] = e.get("pass", {}).get("end_location")
            elif t == "Carry":
                small["end_location"] = e.get("carry", {}).get("end_location")
            elif t == "Shot":
                small["xg"] = e.get("shot", {}).get("statsbomb_xg", 0.0)

            all_events.append(small)

        del events

    if out_jsonl:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, out_jsonl)
        with open(path, "w", encoding="utf-8") as f:
            for event in all_events:
                f.write(json.dumps(event) + "\n")

    return all_events


def load_match_xt_events(db_path: str, match_id: int):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = """
        SELECT
            type,
            player_id,
            player,
            location,
            pass_end_location,
            carry_end_location,
            shot_statsbomb_xg
        FROM events
        WHERE match_id = ?
        AND type IN ('Pass', 'Carry', 'Shot')
        ORDER BY index_num;
    """

    rows = cur.execute(query, (match_id,)).fetchall()
    conn.close()

    xt_events = []

    for t, player_id, player_name, loc, pass_end, carry_end, xg in rows:
        try:
            loc = json.loads(loc) if loc else None
        except:
            loc = None

        if t == "Pass":
            try:
                end_loc = json.loads(pass_end) if pass_end else None
            except:
                end_loc = None

            xt_events.append(
                {
                    "type": "Pass",
                    "player_id": player_id,
                    "player": player_name,
                    "location": loc,
                    "end_location": end_loc,
                }
            )

        elif t == "Carry":
            try:
                end_loc = json.loads(carry_end) if carry_end else None
            except:
                end_loc = None

            xt_events.append(
                {
                    "type": "Carry",
                    "player_id": player_id,
                    "player": player_name,
                    "location": loc,
                    "end_location": end_loc,
                }
            )

        elif t == "Shot":
            xt_events.append(
                {
                    "type": "Shot",
                    "player_id": player_id,
                    "player": player_name,
                    "location": loc,
                    "xg": float(xg) if xg is not None else 0.0,
                }
            )

    return xt_events


if __name__ == "__main__":
    # xg
    # Germany league 2023/2024
    statsbomb_load_league(
        "9/281.json",
        out_csv="shots_training_data_Germany.csv",
        out_dir="output",
    )
    # France league 2022/2023
    statsbomb_load_league(
        "7/235.json", out_csv="shots_training_data_France.csv", out_dir="output"
    )
    # Spain league 2020/2021
    statsbomb_load_league(
        "11/90.json", out_csv="shots_training_data_Spain.csv", out_dir="output"
    )
    # Italy league 2015/2016
    statsbomb_load_league(
        "12/27.json", out_csv="shots_training_data_Italy.csv", out_dir="output"
    )
    # England league 2015/2016
    statsbomb_load_league(
        "2/27.json", out_csv="shots_training_data_England.csv", out_dir="output"
    )
    # uefa
    df_shots = extract_shots_from_db("data/statsbomb_euro2020.db")
    df_shots.to_csv(
        "output/shots_training_data_uefa.csv", index=False, encoding="utf-8-sig"
    )
    # xt
    # Germany league 2023/2024
    statsbomb_load_league_xt(
        "9/281.json", out_jsonl="Germany_xt.jsonl", out_dir="output"
    )
    # France league 2022/2023
    statsbomb_load_league_xt(
        "7/235.json", out_jsonl="France_xt.jsonl", out_dir="output"
    )
    # Spain league 2020/2021
    statsbomb_load_league_xt("11/90.json", out_jsonl="Spain_xt.jsonl", out_dir="output")
    # Italy league 2015/2016
    statsbomb_load_league_xt("12/27.json", out_jsonl="Italy_xt.jsonl", out_dir="output")
    # England league 2015/2016
    statsbomb_load_league_xt(
        "2/27.json", out_jsonl="England_xt.jsonl", out_dir="output"
    )
    # events = load_match_xt_events("data/statsbomb_euro2020.db", 3794686)
    # with open("3794686xt_events.json", "w") as f:
    #     json.dump(events, f)
