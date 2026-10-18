# Music Album Rating Predictor

**Predict a 0–10 album rating from descriptive tags, and compare linear regression,
polynomial interactions, and a neural network without leaking artists across data splits.**

This is a runnable portfolio implementation, not a pretrained commercial music model.
The included 600-row dataset contains **fictional albums and synthetic ratings**. It lets
anyone exercise the full workflow immediately, without scraping or an API key. Real-world
prediction quality has **not** been established.

## Quick start

Requires Python 3.11–3.13. Run from this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
album-ratings train --data data/demo_albums.csv --data-kind synthetic
album-ratings predict --tags 'ambient|atmospheric|instrumental'
uvicorn album_ratings.api:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` for the interactive demo or `/docs` for the API.
`pytest -q` runs the tests. Training is explicit; API startup never silently retrains.

## What it does

1. Validates CSV columns, unique album IDs, nonempty tags, and finite 0–10 ratings.
2. Splits by **artist** into approximately 60% training, 20% validation, and 20% test groups.
3. Fits binary tag features inside each model's training-only pipeline.
4. Compares a mean baseline, ordinary linear regression, regularized degree-two
   polynomial regression (pairwise interactions), and a two-hidden-layer MLP.
5. Selects by validation MAE, refits on train + validation, and evaluates the winner once
   on the held-out test split. A development-set mean baseline is also scored on test.
6. Saves the fitted pipeline, dataset hash, split manifest, metrics, and per-album errors.

```text
CSV → validation → artist-disjoint split → train candidate pipelines
                                            ↓
                                   validation MAE selection
                                            ↓
                               refit winner → held-out test
                                            ↓
                                saved pipeline → API / UI
```

The network is an actual `MLPRegressor` with hidden layers, not a renamed linear model.
Polynomial features are capped at 64 input tags to bound expansion. No artist names,
ratings, or album titles are used as input features. Predictions are clipped to [0, 10]
both at evaluation and inference. Unknown tags are reported; all-unknown requests are rejected.

## Bring your own licensed dataset

```csv
album_id,artist,album,tags,rating
album-001,Example Artist,Example Album,ambient|electronic|atmospheric,7.6
```

Use at least 50 records and 10 distinct artists. Tags are pipe-delimited; multiword tags
remain intact. Convert other rating scales to 0–10 before import. Then run:

```bash
album-ratings train --data data/my_albums.csv --data-kind user_supplied --output artifacts/real
MODEL_PATH=artifacts/real/model.joblib uvicorn album_ratings.api:create_app --factory
```

Do not use review-derived tags that reveal the target rating, mix alternate editions
across splits, or tune models repeatedly against the final test set. Artist aliases and
collaborations require canonicalization beyond this project's case-folding. A real dataset
should record rating provenance, collection time, license, vote counts, and tag availability.

## Project map

| Path | Purpose |
|---|---|
| `src/album_ratings/data.py` | Schema checks and group-aware splits |
| `src/album_ratings/models.py` | Four model pipelines |
| `src/album_ratings/training.py` | Selection, evaluation, artifacts |
| `src/album_ratings/inference.py` | Validated predictions and unseen-tag reporting |
| `src/album_ratings/api.py` | FastAPI endpoints and browser demo |
| `scripts/make_demo_data.py` | Reproduce the fictional dataset |
| `tests/` | Model, leakage, schema, inference, and API tests |
| `reports/` | Actual local test results and synthetic-data run report |

## Limitations and security

The generated ratings contain deliberately learnable patterns. Their metrics are only a
smoke test, **not evidence of accuracy on real albums**. Taste is subjective and this model
has no calibrated confidence interval. The service is a local demo, without authentication
or rate limiting. Do not expose it directly to the internet. Only load model artifacts you
created and trust: `joblib` uses pickle semantics and can execute code on load. Do not commit
private datasets, secrets, or generated artifacts. Dependency ranges should be locked in
your deployment after reviewing the resolved versions.

## References

- [scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: group-aware cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
- [FastAPI](https://fastapi.tiangolo.com/)
