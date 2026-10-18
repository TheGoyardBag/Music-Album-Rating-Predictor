"""CSV validation and group-aware dataset splitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REQUIRED = ("album_id", "artist", "album", "tags", "rating")


def normalize_tags(value: str) -> list[str]:
    """Keep multiword tags intact; accept a pipe-delimited string, not tokenized prose."""
    if not isinstance(value, str):
        raise ValueError("tags must be a pipe-delimited string")
    return sorted({part.strip().lower() for part in value.split("|") if part.strip()})


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={key: str for key in REQUIRED if key != "rating"})
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, REQUIRED].copy()
    if frame.isna().any().any():
        raise ValueError("Missing values are not allowed")
    for key in REQUIRED[:-1]:
        frame[key] = frame[key].str.strip()
        if (frame[key].str.len() == 0).any():
            raise ValueError(f"Empty {key} values are not allowed")
    if frame.album_id.duplicated().any():
        raise ValueError("album_id values must be unique")
    # Artist matching is case-insensitive to avoid accidental group leakage.
    frame["artist"] = frame.artist.str.casefold()
    frame["tags"] = frame.tags.map(lambda x: "|".join(normalize_tags(x)))
    if (frame.tags.str.len() == 0).any() or (frame.tags.str.len() > 4000).any():
        raise ValueError("Each tags field must contain 1–4000 characters")
    frame["rating"] = pd.to_numeric(frame.rating, errors="raise")
    if not np.isfinite(frame.rating).all() or not frame.rating.between(0, 10).all():
        raise ValueError("Ratings must be finite numbers between 0 and 10")
    if len(frame) < 50 or frame.artist.nunique() < 10:
        raise ValueError("Use at least 50 albums from at least 10 distinct artists")
    return frame


def split_by_artist(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Approximately 60/20/20 by artist, not by individual album."""
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    dev_idx, test_idx = next(first.split(frame, groups=frame.artist))
    dev = frame.iloc[dev_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_idx, val_idx = next(second.split(dev, groups=dev.artist))
    return {
        "train": dev.iloc[train_idx].reset_index(drop=True),
        "validation": dev.iloc[val_idx].reset_index(drop=True),
        "test": frame.iloc[test_idx].reset_index(drop=True),
    }
