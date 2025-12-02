# trains xG model
# run from root:
# python -m scripts.compute_xG_LR

from metrics.data_processing import extract_shots_from_db
from metrics.xg import xg_train_model, get_xG_preds

if __name__ == "__main__":
    db_path = "data/statsbomb_euro2020.db"
    pipe = xg_train_model(db_path)

    # sanity check
    df = extract_shots_from_db(db_path)
    preds = get_xG_preds(pipe, df)
    print(preds)
