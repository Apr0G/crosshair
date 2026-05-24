"""
Win-probability model (LightGBM, binary classification).

Predicts P(CT wins this round | current state).

Training data is every row in `round_states` (~1.2 M samples). The split is by
match_id (GroupShuffleSplit) — splitting randomly across rows would leak future
state from a round into training and inflate AUC dramatically.

Realistic generalisation: AUC ≈ 0.90 on held-out matches.
"""
import argparse
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH    = Path(__file__).parent.parent / "data" / "crosshair.db"
MODEL_PATH = Path(__file__).parent.parent / "data" / "win_prob.lgb"

FEATURES = [
    "time_into_round_s",
    "time_remaining_s",
    "post_plant",
    "alive_ct",
    "alive_t",
    "total_hp_ct",
    "total_hp_t",
    "total_armor_ct",
    "total_armor_t",
    "helmets_ct",
    "helmets_t",
    "has_defuser",
    "equip_value_ct",
    "equip_value_t",
    "smokes_ct",
    "smokes_t",
    "flashes_ct",
    "flashes_t",
    "he_ct",
    "he_t",
    "molotovs_ct",
    "molotovs_t",
    "active_smokes",
    "active_infernos",
    "min_dist_ct_to_bomb",  # null pre-plant, lgbm handles it
    "min_dist_t_to_bomb",
    "ct_spotted_count",
    "t_spotted_count",
    "ct_heard_enemy",
    "t_heard_enemy",
    "ct_spread",
    "t_spread",
    "map",
]

TARGET = "round_won_ct"


def load_training_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            f"SELECT match_id, {', '.join(FEATURES)}, {TARGET} FROM round_states",
            conn,
        )
    finally:
        conn.close()
    df["map"] = df["map"].astype("category")
    return df


def train(eval: bool = False):
    import lightgbm as lgb
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, log_loss

    df = load_training_data()
    print(f"loaded {len(df):,} samples from {df['match_id'].nunique()} matches, {df['map'].nunique()} maps")
    print(f"ct win rate: {df[TARGET].mean():.3f}")

    X = df[FEATURES]
    y = df[TARGET]
    groups = df["match_id"]

    # Split by match — same-round samples must stay in the same fold or the model
    # leaks future round state into training and overstates AUC.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    print(f"train: {len(X_train):,} samples / val: {len(X_val):,} samples")

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=["map"], free_raw_data=False)
    val_set   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=["map"], reference=train_set, free_raw_data=False)

    params = {
        "objective":        "binary",
        "metric":           ["binary_logloss", "auc"],
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":          -1,
        "n_jobs":           -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)],
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"\nmodel saved → {MODEL_PATH}")

    if eval:
        preds = model.predict(X_val)
        print(f"\nauc:      {roc_auc_score(y_val, preds):.4f}")
        print(f"log-loss: {log_loss(y_val, preds):.4f}")

        imp = pd.Series(model.feature_importance(importance_type="gain"), index=FEATURES)
        print("\ntop features:")
        print(imp.sort_values(ascending=False).head(10).to_string())

    return model


_model = None

def load_model():
    global _model
    if _model is None:
        import lightgbm as lgb
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"no model at {MODEL_PATH} — train first")
        _model = lgb.Booster(model_file=str(MODEL_PATH))
    return _model


def predict(state: dict) -> float:
    return float(load_model().predict(_state_to_row(state))[0])


def predict_batch(states: list[dict]) -> np.ndarray:
    rows = pd.concat([_state_to_row(s) for s in states], ignore_index=True)
    return load_model().predict(rows)


def _state_to_row(state: dict) -> pd.DataFrame:
    row = {f: state.get(f, np.nan) for f in FEATURES}
    row["map"] = str(row.get("map") or "unknown")
    df = pd.DataFrame([row])
    df["map"] = df["map"].astype("category")
    # None becomes object dtype in single-row DataFrames; cast numeric cols to float
    for col in df.columns:
        if col != "map" and df[col].dtype == object:
            df[col] = df[col].astype(float)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval",  action="store_true")
    args = parser.parse_args()

    if args.train:
        train(eval=args.eval)
    else:
        parser.print_help()
