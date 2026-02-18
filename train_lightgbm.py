import os
import json
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


DATA_PATH = "training_dataset.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TARGETS = ["y_7", "y_14"]

# ---- time split (70/15/15) ----
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
TEST_FRAC  = 0.15

SEED = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_model():
    return LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1
    )



def add_lag_features(df):
    """
    Add simple lag + rolling features PER AOI.
    This usually helps forecasting a lot.
    """
    df = df.sort_values(["aoi_id", "date"]).copy()

    # lags in days relative to your sampling
    # (you used step_days=5 earlier, so lag_1 = previous sample ~5 days)
    df["stress_lag_1"] = df.groupby("aoi_id")["stress_mean"].shift(1)
    df["stress_lag_2"] = df.groupby("aoi_id")["stress_mean"].shift(2)
    df["stress_lag_3"] = df.groupby("aoi_id")["stress_mean"].shift(3)

    df["stress_roll_3"] = (
        df.groupby("aoi_id")["stress_mean"]
        .shift(1)
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def time_split_indices(df_dates):
    """
    df_dates is already sorted by date.
    """
    n = len(df_dates)
    n_train = int(n * TRAIN_FRAC)
    n_valid = int(n * VALID_FRAC)
    n_test = n - n_train - n_valid

    train_idx = np.arange(0, n_train)
    valid_idx = np.arange(n_train, n_train + n_valid)
    test_idx  = np.arange(n_train + n_valid, n)

    return train_idx, valid_idx, test_idx


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded: {DATA_PATH}")
    print(f"[INFO] Rows total: {len(df)}")

    # Required columns check (adjust if your column names differ)
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column in training_dataset.csv")
    if "aoi_id" not in df.columns:
        raise ValueError("Missing 'aoi_id' column in training_dataset.csv")
    if "stress_mean" not in df.columns:
        raise ValueError("Missing 'stress_mean' column in training_dataset.csv")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Add lag/rolling features
    df = add_lag_features(df)

    # Feature columns: numeric only, excluding targets
    drop_cols = set(["date"] + TARGETS)
    feat_cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if not c.startswith("pix_"):
                feat_cols.append(c)


    print(f"[INFO] Feature count: {len(feat_cols)}")

    # Global split by date order (simple and strict)
    # (Alternative is split by unique dates; but this is fine for now.)
    train_idx, valid_idx, test_idx = time_split_indices(df["date"])
    df_train_all = df.iloc[train_idx].copy()
    df_valid_all = df.iloc[valid_idx].copy()
    df_test_all  = df.iloc[test_idx].copy()

    print(f"[INFO] Train rows: {len(df_train_all)}  Valid rows: {len(df_valid_all)}  Test rows: {len(df_test_all)}")
    print(f"[INFO] Train date range: {df_train_all['date'].min().date()} -> {df_train_all['date'].max().date()}")
    print(f"[INFO] Valid date range: {df_valid_all['date'].min().date()} -> {df_valid_all['date'].max().date()}")
    print(f"[INFO] Test  date range: {df_test_all['date'].min().date()} -> {df_test_all['date'].max().date()}")

    all_metrics = {}

    for target in TARGETS:
        # keep only rows where target is available
        tr = df_train_all.dropna(subset=[target]).copy()
        va = df_valid_all.dropna(subset=[target]).copy()
        te = df_test_all.dropna(subset=[target]).copy()

        print(f"\n[TRAIN] {target}: train={len(tr)} valid={len(va)} test={len(te)}")

        # Also drop rows where lag features are NaN (first samples per AOI)
        needed = feat_cols + [target]
        tr = tr.dropna(subset=needed)
        va = va.dropna(subset=needed)
        te = te.dropna(subset=needed)

        print(f"[TRAIN] {target} after lag-NaN drop: train={len(tr)} valid={len(va)} test={len(te)}")

        X_train, y_train = tr[feat_cols], tr[target]
        X_valid, y_valid = va[feat_cols], va[target]
        X_test,  y_test  = te[feat_cols], te[target]

        model = make_model()

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse"
        )


        pred_tr = model.predict(X_train)
        pred_va = model.predict(X_valid) if len(X_valid) else np.array([])
        pred_te = model.predict(X_test)  if len(X_test)  else np.array([])

        m_tr = metrics(y_train, pred_tr)
        m_va = metrics(y_valid, pred_va) if len(pred_va) else None
        m_te = metrics(y_test, pred_te)  if len(pred_te) else None

        print(f"[RESULT] {target} Train RMSE={m_tr['RMSE']:.4f}  MAE={m_tr['MAE']:.4f}  R2={m_tr['R2']:.4f}")
        if m_va:
            print(f"[RESULT] {target} Valid RMSE={m_va['RMSE']:.4f}  MAE={m_va['MAE']:.4f}  R2={m_va['R2']:.4f}")
        if m_te:
            print(f"[RESULT] {target} Test  RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}  R2={m_te['R2']:.4f}")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"stress_{target}.joblib")
        joblib.dump(model, model_path)
        print(f"[SAVE] Saved model: {model_path}")

        # Save feature importance
        fi = pd.DataFrame({
            "feature": feat_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi_path = os.path.join(MODELS_DIR, f"feature_importance_{target}.csv")
        fi.to_csv(fi_path, index=False)
        print(f"[SAVE] Saved feature importance: {fi_path}")

        all_metrics[target] = {"train": m_tr, "valid": m_va, "test": m_te}

    # Save metrics json
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SAVE] Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
