import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from metrics.data_processing import extract_shots_from_db, statsbomb_load_league

XG_FEATURES = [
    "distance",
    "angle",
    "shot_body_part",
    "shot_technique",
    "shot_type",
    "play_pattern",
    "num_teammates",
    "num_opponents",
]


# train and return the model on the statsbomb uefa 2020 data and statsbomb github open data
def xg_train_model(db_path: str = "data/statsbomb_euro2020.db"):
    """
    Trains and returns a logistic regression model on the given database

    Parameters
    ----------
    db_path : str
        A string with the path to the database file.
        The database should have shot data, and we have made this function to work with the
        statsbomb uefa euro 2020 dataset (it might not generalize)

    Returns
    -------
    pipe : sklearn.pipeline.Pipeline
        the trained model with the full pipeline (encoder + logistic regression)
    """

    # get the shots
    # Germany league 2023/2024
    df_1 = statsbomb_load_league("9/281.json")
    # France league 2022/2023
    df_2 = statsbomb_load_league("7/235.json")
    # Spain league 2020/2021
    df_3 = statsbomb_load_league("11/90.json")
    # Italy league 2015/2016
    df_4 = statsbomb_load_league("12/27.json")
    # England league 2015/2016
    df_5 = statsbomb_load_league("2/27.json")
    # uefa 2020
    df_uefa2020 = extract_shots_from_db(db_path)

    shots_df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_uefa2020])

    # prepare data
    X = shots_df[XG_FEATURES]
    y = shots_df["goal"]

    numeric_features = [
        "distance",
        "angle",
        "num_teammates",
        "num_opponents",
    ]
    categorical_features = [
        "shot_body_part",
        "shot_technique",
        "shot_type",
        "play_pattern",
    ]

    # set up model
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # just using logistic regression classifier
    lr_classifier = LogisticRegression(max_iter=2000)

    pipe = Pipeline([("prep", preprocessor), ("clf", lr_classifier)])

    # train on full dataset since it is expected to estimate xG for new matches
    pipe.fit(X, y)

    os.makedirs("metrics/weights", exist_ok=True)
    with open("metrics/weights/xg_pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)

    return pipe


def load_model(model_path: str) -> Pipeline:
    with open(model_path, "rb") as f:
        pipeline: Pipeline = pickle.load(f)

        return pipeline


def get_xG_preds(pipeline, dataframe):
    """
    Returns xG predictions, one per row in the dataframe.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A trained pipeline for the given dataframe (output of xg_train_model)
    dataframe : pandas.DataFrame
        The dataframe with the correct column names for the pipeline.
        You might need to call extract_shots_from_db or just make this function better lol

    Returns
    -------
    preds : idk, maybe a list?
        The xG predictions (probabilities)
    """
    preds = pipeline.predict_proba(dataframe)[:, 1]
    return preds
