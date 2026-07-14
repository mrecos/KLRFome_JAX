"""Tests for the tabular mean-embedding helpers, incl. the RBF-on-embeddings kernel."""

import numpy as np
import jax.random as random
import pytest

from klrfome.data.formats import SampleCollection
from klrfome.data.tabular import (
    fit_mean_embedding,
    mean_embedding_predict,
    mean_embedding_heldout,
    predict_xy_surface,
)


def _toy_bags(n_per=6, m=12, d=4, sep=1.0, seed=0):
    bags = []
    for i in range(n_per):
        bags.append(
            SampleCollection(random.normal(random.PRNGKey(seed + i), (m, d)) + sep, 1, f"s{i}")
        )
    for i in range(n_per):
        bags.append(
            SampleCollection(
                random.normal(random.PRNGKey(seed + 100 + i), (m, d)) - sep, 0, f"b{i}"
            )
        )
    return bags


@pytest.mark.parametrize("mode", ["linear", "rbf"])
def test_mean_embedding_kernel_modes_run(mode):
    """Both the linear and RBF-on-embeddings kernels fit, predict, and score held-out
    bags with valid probabilities/AUC."""
    bags = _toy_bags()
    train, test = bags[:5] + bags[6:11], [bags[5], bags[11]]

    model = fit_mean_embedding(train, sigma=2.0, embedding_kernel=mode, n_features=128, seed=0)
    assert model["embedding_kernel"] == mode
    if mode == "rbf":
        assert model["embedding_sigma"] is not None and model["embedding_sigma"] > 0
    else:
        assert model["embedding_sigma"] is None

    p = mean_embedding_predict(model, test)
    assert p.shape == (2,)
    assert np.all((p >= 0) & (p <= 1))

    auc, probs, y = mean_embedding_heldout(train, test, sigma=2.0, embedding_kernel=mode, seed=0)
    assert 0.0 <= auc <= 1.0
    assert len(probs) == len(y) == 2


def test_rbf_embedding_sigma_is_passthrough():
    """An explicit embedding_sigma is respected (not overwritten by calibration)."""
    bags = _toy_bags()
    model = fit_mean_embedding(bags, sigma=2.0, embedding_kernel="rbf", embedding_sigma=3.0, seed=0)
    assert model["embedding_sigma"] == 3.0


def test_predict_xy_surface_rbf():
    """predict_xy_surface must honour the RBF-on-embeddings model (not assume linear)."""
    bags = _toy_bags()
    model = fit_mean_embedding(bags, sigma=2.0, embedding_kernel="rbf", n_features=64, seed=0)
    rng = np.random.default_rng(0)
    cell_xy = rng.normal(size=(200, 2))
    cell_X = rng.normal(size=(200, 4))
    probs = predict_xy_surface(model, cell_xy[:50], cell_xy, cell_X, k=8)
    assert probs.shape == (50,)
    assert np.all((probs >= 0) & (probs <= 1))
