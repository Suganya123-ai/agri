import numpy as np
from datetime import datetime, timedelta

from agri_core import (
    process_sentinel2_data,
    load_datasets_from_files,
    _compute_aoi_current_stress,
)

def split_into_weeks(date_range):
    start_str, end_str = date_range.split("/")
    start = datetime.fromisoformat(start_str).date()
    end = datetime.fromisoformat(end_str).date()

    weeks = []
    current = start

    while current < end:
        week_end = min(current + timedelta(days=7), end)
        weeks.append((current.isoformat(), week_end.isoformat()))
        current = week_end

    return weeks


def compute_weekly_series(bbox, date_range, max_cloud):
    weekly_ranges = split_into_weeks(date_range)

    images = []
    stress_series = []

    for start, end in weekly_ranges:
        try:
            files = process_sentinel2_data(
                bbox=bbox,
                date_range=f"{start}/{end}",
                max_cloud=max_cloud
            )

            datasets = load_datasets_from_files(files)
            stress = _compute_aoi_current_stress(datasets)

            quicklook = (
                files.get("Stress_Analysis")
                or files.get("True_Color")
                or files.get("Stress_Score")
            )

            stress_value = None
            if stress is not None and np.isfinite(stress):
                stress_value = float(stress)

            images.append({
                "date": start,
                "stress": float(stress) if np.isfinite(stress) else None,
                "files": files
            })

            if stress_value is not None:
                stress_series.append({
                    "date": start,
                    "stress": stress_value
                })

        except Exception as e:
            print("week skipped:", e)

    return images, stress_series