import pandas as pd


def build_live_row_features(df_history: pd.DataFrame, aoi_id: str, date, base_features: dict):
    """
    df_history must contain columns: aoi_id, date, stress_mean
    base_features: the features you compute in viewer for current date (weather, indices, etc.)

    This function adds the lag/rolling features needed by the trained model.
    """

    df = df_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    date = pd.to_datetime(date)

    # history for this AOI BEFORE current date
    h = df[(df["aoi_id"] == aoi_id) & (df["date"] < date)].sort_values("date")

    # take last 3 stress values
    last = h["stress_mean"].tail(3).tolist()  # could be 0..3 values

    # default NaNs if not enough history
    stress_lag_1 = last[-1] if len(last) >= 1 else None
    stress_lag_2 = last[-2] if len(last) >= 2 else None
    stress_lag_3 = last[-3] if len(last) >= 3 else None

    # rolling mean over previous 3 (excluding current)
    stress_roll_3 = sum(last) / len(last) if len(last) > 0 else None

    row = dict(base_features)
    row["stress_lag_1"] = stress_lag_1
    row["stress_lag_2"] = stress_lag_2
    row["stress_lag_3"] = stress_lag_3
    row["stress_roll_3"] = stress_roll_3

    return row
