# src/utils.py
# Helper Functions for synthetic ED dataset generation

import os
import random
from datetime import datetime, timedelta, time
import pandas as pd

# MIMIC-like datetime string format
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------
# Random timestamp helpers
# ---------------------------------------------------------
def random_arrival_time(lookback_days: int = 30) -> datetime:
    """
    Random date in the last 'lookback_days' with random time-of-day.
    Example: used for first admission arrival timestamp.
    """
    base_date = datetime.now().date() - timedelta(days=random.randint(1, lookback_days))
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime.combine(base_date, time(hour, minute, second))


def random_time_after(start: datetime, min_minutes: int, max_minutes: int) -> datetime:
    """
    Return a datetime after 'start' with random minutes added.
    NOTE: kept for backward compatibility (older code).
    """
    return start + timedelta(minutes=random.randint(min_minutes, max_minutes))


def sample_uniform_minutes(min_minutes: int, max_minutes: int) -> timedelta:
    """
    Sample a timedelta uniformly between min_minutes and max_minutes.
    Used for Option A: initial assessment delay, ICU start delay, etc.
    """
    return timedelta(minutes=random.randint(min_minutes, max_minutes))


# ---------------------------------------------------------
# Demographics helpers
# ---------------------------------------------------------
def derive_age_group(age: int) -> str:
    """Map age_years to an age band."""
    if age <= 18:
        return "0-18"
    elif age <= 40:
        return "19-40"
    elif age <= 65:
        return "41-65"
    return "65+"


# ---------------------------------------------------------
# Inter-admission gap helper (Gaussian)
# (You may still use this in older scripts, but we now
# prefer empirical gaps from MIMIC where possible.)
# ---------------------------------------------------------
def sample_gap_hours(mean_h: float, std_h: float) -> float:
    """
    Sample a positive inter-admission gap in hours.
    Uses Gaussian sampling; clamps to >= 1 hour to avoid overlaps.
    """
    if std_h is None or std_h == 0:
        return max(1.0, mean_h)
    gap = random.gauss(mean_h, std_h)
    return max(1.0, gap)


# ---------------------------------------------------------
# Empirical distribution utilities (MIMIC-derived)
# ---------------------------------------------------------
def load_gap_series(path: str, col: str) -> pd.Series:
    """
    Load a numeric gap column from a CSV (e.g., gap_hours, gap_minutes, gap_days),
    drop invalid/negative values, return as a pandas Series.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gap file not found: {path}")

    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in {path}. Found: {list(df.columns)}")

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        raise ValueError(f"No positive values found in {path} column '{col}'")
    return s


def sample_empirical(series: pd.Series, min_val: float = 0.0) -> float:
    """
    Sample a value from an empirical distribution (resampling with replacement).
    min_val ensures we don't generate unrealistically tiny gaps.
    """
    if series is None or len(series) == 0:
        raise ValueError("Empirical series is empty or None.")

    while True:
        v = float(series.sample(1).iloc[0])
        if v >= min_val:
            return v


# ---------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------
def format_datetime_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert datetime columns to MIMIC-like string format: YYYY-MM-DD HH:MM:SS.
    This makes your synthetic CSV look consistent with MIMIC exports.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(DATETIME_FMT)
    return df
