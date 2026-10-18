"""Select on validation MAE; refit on development data; evaluate the winner once on test."""
from __future__ import annotations
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data import load_data, split_by_artist
from .models import build_models


def metrics(actual, predicted) -> dict[str, float]:
    # Apply the same range constraint during evaluation and inference.
    predicted = np.clip(np.asarray(predicted, dtype=float), 0.0, 10.0)
    if not np.isfinite(predicted).all():
        raise ValueError("The estimator produced a non-finite prediction")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def train(csv: str | Path, output: str | Path, seed: int = 42,
          max_tags: int = 40, data_kind: str = "user_supplied") -> dict:
    if data_kind not in {"synthetic", "user_supplied"}:
        raise ValueError("data_kind must be synthetic or user_supplied")
    csv, output = Path(csv), Path(output)
    frame = load_data(csv)
    splits = split_by_artist(frame, seed)
    models = build_models(seed, max_tags)
    validation, training_warnings = {}, {}
    for name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(splits["train"].tags, splits["train"].rating)
            validation[name] = metrics(splits["validation"].rating,
                                       model.predict(splits["validation"].tags))
        training_warnings[name] = sorted({str(item.message) for item in caught})
    winner = min(validation, key=lambda name: validation[name]["mae"])
    development = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    final = clone(models[winner])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final.fit(development.tags, development.rating)
    training_warnings["refit"] = sorted({str(item.message) for item in caught})
    predicted = np.clip(final.predict(splits["test"].tags), 0, 10)
    baseline_prediction = np.repeat(development.rating.mean(), len(splits["test"]))
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_kind": data_kind,
        "dataset_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "seed": seed, "max_tags": max_tags, "rating_scale": [0, 10],
        "split_method": "artist-disjoint train/validation/test; ~60/20/20 by artist",
        "splits": {k: {"albums": len(v), "artists": v.artist.nunique()} for k, v in splits.items()},
        "validation": validation,
        "selected_model": winner,
        "test": metrics(splits["test"].rating, predicted),
        "test_mean_baseline": metrics(splits["test"].rating, baseline_prediction),
        "warnings": training_warnings,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "limitations": [
            "Synthetic data metrics are a software smoke test, not evidence of real music prediction.",
            "Ratings are subjective; no causal or quality judgment is implied.",
            "Tags may encode post-release reviews; evaluate temporal leakage on real data.",
            "Predictions are clipped to [0, 10]; there is no calibrated uncertainty estimate.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "report": report}, output / "model.joblib")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = splits["test"][["album_id", "artist", "album", "rating"]].copy()
    rows["prediction"] = predicted
    rows["absolute_error"] = abs(rows.rating - predicted)
    rows.to_csv(output / "test_predictions.csv", index=False)
    (output / "split_manifest.json").write_text(json.dumps(
        {key: list(value.album_id) for key, value in splits.items()}, indent=2) + "\n")
    return report
