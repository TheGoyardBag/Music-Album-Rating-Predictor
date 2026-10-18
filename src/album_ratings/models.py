"""Every learned transform is fitted inside a scikit-learn pipeline."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from .data import normalize_tags


def build_models(seed: int = 42, max_tags: int = 40) -> dict[str, Pipeline]:
    if not 2 <= max_tags <= 64:
        raise ValueError("max_tags must be between 2 and 64 to bound polynomial feature growth")
    def features():
        return CountVectorizer(
            tokenizer=normalize_tags, token_pattern=None, lowercase=False,
            binary=True, max_features=max_tags, dtype=np.float64,
        )
    # Degree two is bounded: 64 input tags produce at most 2,144 features.
    # With binary inputs, squared terms duplicate inputs, so use pairwise interactions only.
    return {
        "mean_baseline": Pipeline([("tags", features()), ("regressor", DummyRegressor())]),
        "linear": Pipeline([("tags", features()), ("regressor", LinearRegression())]),
        "polynomial": Pipeline([
            ("tags", features()),
            ("interactions", PolynomialFeatures(degree=2, include_bias=False,
                                               interaction_only=True)),
            ("regressor", Ridge(alpha=10.0, solver="lsqr")),
        ]),
        "neural_network": Pipeline([
            ("tags", features()),
            ("scale", StandardScaler(with_mean=False)),
            ("regressor", TransformedTargetRegressor(
                regressor=MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0,
                    max_iter=600, random_state=seed, early_stopping=False,
                    learning_rate_init=0.003),
                transformer=StandardScaler(),
            )),
        ]),
    }
