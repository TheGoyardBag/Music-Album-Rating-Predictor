from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from album_ratings.data import load_data, normalize_tags, split_by_artist
from album_ratings.models import build_models
from album_ratings.training import train, metrics
from album_ratings.inference import Predictor
from album_ratings.api import create_app

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def frame():
    return load_data(ROOT / "data/demo_albums.csv")

@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp("trained")
    report = train(ROOT / "data/demo_albums.csv", folder, data_kind="synthetic")
    return folder, report


def test_normalizes_whole_tags():
    assert normalize_tags(" Dream Pop |warm|dream pop||") == ["dream pop", "warm"]


def test_disjoint_artist_splits(frame):
    parts = split_by_artist(frame)
    sets = [set(part.artist) for part in parts.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(len(part) for part in parts.values()) == len(frame)


def test_split_reproducibility(frame):
    assert split_by_artist(frame)["test"].equals(split_by_artist(frame)["test"])


def test_validation_only_tag_is_not_learned():
    model = build_models()["linear"]
    model.fit(["rock|warm", "jazz|warm", "rock|energetic"], [6, 7, 8])
    model.predict(["secret-validation-tag"])
    assert "secret-validation-tag" not in model.named_steps["tags"].vocabulary_


@pytest.mark.parametrize("model_name", ["mean_baseline", "linear", "polynomial", "neural_network"])
def test_each_model_predicts_finite(frame, model_name):
    model = build_models()[model_name]
    model.fit(frame.tags[:100], frame.rating[:100])
    assert np.isfinite(model.predict(frame.tags[100:103])).all()


@pytest.mark.parametrize("kind", ["missing", "duplicate", "bad_rating", "nan_rating", "empty_tags", "few_artists"])
def test_bad_csv_is_rejected(frame, tmp_path, kind):
    data = frame.copy()
    if kind == "missing": data = data.drop(columns="tags")
    elif kind == "duplicate": data.loc[1, "album_id"] = data.loc[0, "album_id"]
    elif kind == "bad_rating": data.loc[0, "rating"] = 11
    elif kind == "nan_rating": data.loc[0, "rating"] = np.nan
    elif kind == "empty_tags": data.loc[0, "tags"] = "||"
    else: data["artist"] = "single artist"
    path = tmp_path / "bad.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError): load_data(path)


def test_report_and_artifacts(trained):
    folder, report = trained
    assert report["dataset_kind"] == "synthetic"
    assert report["selected_model"] == min(report["validation"], key=lambda n:report["validation"][n]["mae"])
    assert all((folder / name).exists() for name in
               ["model.joblib", "report.json", "test_predictions.csv", "split_manifest.json"])
    assert 0 <= report["test"]["mae"] <= 10
    ids = json.loads((folder / "split_manifest.json").read_text())
    assert not (set(ids["train"]) & set(ids["test"]))


def test_inference_has_provenance(trained):
    result = Predictor(trained[0] / "model.joblib").predict(["ambient", "unknown-tag"])
    assert result["unknown_tags"] == ["unknown-tag"]
    assert result["dataset_kind"] == "synthetic"
    assert 0 <= result["predicted_rating"] <= 10


@pytest.mark.parametrize("tags", [[], ["unseen-only"], ["ambient|rock"], ["ambient"] * 65])
def test_bad_predictions_rejected(trained, tags):
    with pytest.raises(ValueError): Predictor(trained[0] / "model.joblib").predict(tags)


def test_api(trained):
    with TestClient(create_app(trained[0] / "model.joblib")) as client:
        assert client.get("/health").status_code == 200
        assert "ALBUM ATLAS" in client.get("/").text
        assert client.post("/predict", json={"tags": ["ambient"]}).status_code == 200
        assert client.post("/predict", json={"tags": ["no-such-tag"]}).status_code == 422
        assert client.post("/predict", json={"tags": ["ambient"], "unexpected": True}).status_code == 422


def test_bounds_polynomial_expansion():
    with pytest.raises(ValueError): build_models(max_tags=1000)


def test_clipping_is_part_of_evaluation():
    assert metrics([0, 10], [-100, 100])["mae"] == 0
