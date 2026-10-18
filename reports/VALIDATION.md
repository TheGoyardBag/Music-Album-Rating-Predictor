# Local validation

- Python 3.13.5; scikit-learn 1.8.0; NumPy 2.3.5; pandas 2.2.3; FastAPI 0.128.2.
- **23 pytest cases passed**. Tests include artist-disjoint splits, training-only vocabulary,
  all four estimators, CSV validation, saved artifacts, prediction validation, and FastAPI TestClient requests.
- Editable package installation and the `album-ratings` command-line entrypoint succeeded.
- Full synthetic-data training completed; the validation-selected model was polynomial regression.
- `synthetic_run.json` and the associated CSV/manifest are the real outputs of that local run.
- No real music dataset was supplied. These metrics are not real-world accuracy evidence.
- The local API started successfully. A browser automation attempt was blocked by the execution
  environment, so the browser interface has not received a completed visual/end-to-end check here.
- GitHub Actions is configured, not claimed to have run on GitHub. Other Python versions in the
  workflow matrix have not been exercised in this local environment.

Reproduce with `pytest -q` and the README's training command. Small numerical differences are
possible across environments. Preserve the data hash, split, and package versions for comparisons.
