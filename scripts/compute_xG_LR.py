# trains xG model

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
sys.path.append('../metrics')
from data_processing import extract_shots_from_db

# train and return the model on the statsbomb euro 2020 data
def xg_train_model(db_path:str='../statsbomb/statsbomb_euro2020.db'):
    '''
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
    '''

    # get the shots
    shots_df = extract_shots_from_db(db_path)
    # drop nans
    shots_df = shots_df.dropna(how="any")

    # prepare data
    X = shots_df[["distance", "angle", "body_part", "technique", "shot_type",
            "play_pattern", "num_teammates", "num_opponents"]]
    y = shots_df["goal"]

    numeric_features = ["distance", "angle", "num_teammates", "num_opponents"]
    categorical_features = ["body_part", "technique", "shot_type", "play_pattern"]

    # set up model
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # just using logistic regression classifier
    lr_classifier = LogisticRegression(max_iter=2000)

    pipe = Pipeline([("prep", preprocessor), ("clf", lr_classifier)])
    pipe.fit(X_train, y_train)

    return pipe

def get_xG_preds(pipeline, dataframe):
    '''
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
    '''
    preds = pipeline.predict_proba(dataframe)[:, 1]
    return preds

if __name__ == "__main__":
    db_path = "../statsbomb/statsbomb_euro2020.db"
    pipe = xg_train_model(db_path)

    df = extract_shots_from_db(db_path)
    df = df.dropna(how="any")

    preds = get_xG_preds(pipe, df)

    print(preds)