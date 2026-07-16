"""Random Fourier Features (RFF) approximations for the RBF kernel."""

from functools import lru_cache
import time
from typing import Literal, Optional

import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import jit
from jaxtyping import Array, Float

RFFScheme = Literal["iid", "orthogonal"]


@jit
def _rff_feature_map(
    X: Float[Array, "n d"],
    W: Float[Array, "d m"],
) -> Float[Array, "n D"]:
    """Core RFF map (phase-free sin/cos estimator):

        phi(x) = sqrt(1/m) * [cos(Wx), sin(Wx)]    with m = W.shape[1] frequencies, D = 2m.

    The phase-free sin/cos pair is unbiased for the RBF kernel and has strictly LOWER
    variance than the random-offset ``sqrt(2/D) cos(Wx + b)`` estimator at the same
    feature budget (Sutherland & Schneider, 2015) -- a free accuracy upgrade.

    W is an explicit (traced) argument rather than a closed-over constant, so the
    compiled function always uses the current weights (no stale-weight baking via a
    static ``self``).
    """
    proj = jnp.dot(X, W)  # (n, m)
    m = W.shape[1]
    return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=1) * jnp.sqrt(1.0 / m)


def _orthogonal_frequencies(
    key: Array, input_dim: int, n_frequencies: int, sigma: float
) -> Float[Array, "d m"]:
    """Draw block-orthogonal Gaussian spectral frequencies.

    Each column has the Gaussian spectral distribution required by the RBF
    kernel. Columns within a block have orthogonal directions, reducing
    redundant directions without changing the phase-free feature map.
    """
    n_blocks = (n_frequencies + input_dim - 1) // input_dim
    keys = random.split(key, 2 * n_blocks)
    blocks = []
    for block_index in range(n_blocks):
        gaussian = random.normal(keys[2 * block_index], (input_dim, input_dim))
        orthogonal, triangular = jnp.linalg.qr(gaussian)
        signs = jnp.sign(jnp.diag(triangular))
        signs = jnp.where(signs == 0, 1.0, signs)
        orthogonal = orthogonal * signs[None, :]
        radial_gaussian = random.normal(keys[2 * block_index + 1], (input_dim, input_dim))
        radii = jnp.linalg.norm(radial_gaussian, axis=0)
        blocks.append(orthogonal * radii[None, :])
    return jnp.concatenate(blocks, axis=1)[:, :n_frequencies] / sigma


@lru_cache(maxsize=64)
def _cached_orthogonal_unit_frequencies(
    seed: int, input_dim: int, n_frequencies: int
) -> np.ndarray:
    """Cache bandwidth-free ORF draws for reuse across folds and model variants."""
    values = _orthogonal_frequencies(random.PRNGKey(seed), input_dim, n_frequencies, sigma=1.0)
    output = np.asarray(values, dtype=np.float32)
    output.setflags(write=False)
    return output


def clear_rff_frequency_cache() -> None:
    """Clear cached orthogonal unit-frequency matrices."""
    _cached_orthogonal_unit_frequencies.cache_clear()


def rff_frequency_cache_info() -> dict[str, int]:
    """Return ORF cache counters for benchmarks and diagnostics."""
    info = _cached_orthogonal_unit_frequencies.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": int(info.maxsize or 0),
        "currsize": info.currsize,
    }


class RandomFourierFeatures:
    """
    Random Fourier Features approximation to RBF kernel.

    Approximates: k(x, y) ≈ φ(x)ᵀφ(y)

    Where φ(x) = sqrt(1/m) * [cos(Wx), sin(Wx)]   (phase-free sin/cos estimator)

    This transforms the kernel computation from O(n²) to O(nD) where
    D = 2 * n_features is the output dimension.

    Reference:
        Rahimi & Recht (2007). "Random Features for Large-Scale Kernel Machines"
        Sutherland & Schneider (2015). "On the Error of Random Fourier Features"
        Yu et al. (2016). "Orthogonal Random Features"

    Parameters:
        sigma: Bandwidth of the RBF kernel to approximate
        n_features: Number of random FREQUENCIES m. Higher = better approximation.
            The output dimension is D = 2 * n_features (cos and sin per frequency).
        seed: Random seed for reproducibility
        scheme: ``iid`` for ordinary Gaussian frequencies or ``orthogonal``
            for block-orthogonal frequencies with Gaussian radial scaling.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        n_features: int = 256,
        seed: int = 42,
        scheme: RFFScheme = "iid",
    ):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if scheme not in ("iid", "orthogonal"):
            raise ValueError("scheme must be 'iid' or 'orthogonal'")

        self._sigma = sigma
        self._n_features = n_features
        self._seed = seed
        self._scheme = scheme
        self._W: Optional[Float[Array, "d D"]] = None  # Lazily initialized
        self._b: Optional[Float[Array, "D"]] = None
        self._input_dim: Optional[int] = None
        self._frequency_cache_hit: Optional[bool] = None
        self._frequency_initialization_seconds: Optional[float] = None

    @property
    def sigma(self) -> float:
        """Kernel bandwidth parameter."""
        return self._sigma

    @property
    def n_features(self) -> int:
        """Number of random features."""
        return self._n_features

    @property
    def scheme(self) -> RFFScheme:
        """Frequency construction used by the fitted feature map."""
        return self._scheme

    @property
    def frequency_cache_hit(self) -> Optional[bool]:
        """Whether ORF initialization reused a cached unit-frequency matrix."""
        return self._frequency_cache_hit

    @property
    def frequency_initialization_seconds(self) -> Optional[float]:
        """Wall time spent constructing or restoring the current frequencies."""
        return self._frequency_initialization_seconds

    def _initialize_weights(self, input_dim: int):
        """Initialize random frequencies for the (phase-free) feature map.

        ``n_features`` is the number of random FREQUENCIES m (matching the
        Rahimi-Recht / original-offset convention). The sin/cos estimator emits
        two features -- cos and sin -- per frequency, so the OUTPUT dimension is
        D = 2 * n_features. Keeping the frequency count equal to ``n_features``
        (rather than n_features//2) preserves how finely the mean embedding
        resolves a distribution, which matters for tail metrics like top-area
        lift even though it does not move AUC.
        """
        if self._W is not None and self._input_dim == input_dim:
            return

        started = time.perf_counter()
        key = random.PRNGKey(self._seed)
        m = self._n_features  # number of frequencies; output dim D = 2 * m
        # w ~ N(0, I/sigma^2): the spectral density of the RBF kernel.
        if self._scheme == "iid":
            self._W = random.normal(key, (input_dim, m)) / self._sigma
            self._frequency_cache_hit = None
        else:
            before = _cached_orthogonal_unit_frequencies.cache_info()
            unit_frequencies = _cached_orthogonal_unit_frequencies(self._seed, input_dim, m)
            after = _cached_orthogonal_unit_frequencies.cache_info()
            self._frequency_cache_hit = after.hits > before.hits
            self._W = jnp.asarray(unit_frequencies) / self._sigma
        self._b = None  # no random offset in the sin/cos estimator
        self._input_dim = input_dim
        self._frequency_initialization_seconds = time.perf_counter() - started

    def feature_map(self, X: Float[Array, "n d"]) -> Float[Array, "n D"]:
        """
        Compute random Fourier features.

        φ(x) = sqrt(1/m) * [cos(Wx), sin(Wx)]

        Parameters:
            X: Input points, shape (n, d)

        Returns:
            Feature map of shape (n, D) with D = 2 * n_features
        """
        if self._W is None or self._input_dim != X.shape[1]:
            raise RuntimeError(
                "Weights not initialized. Call _initialize_weights() first, "
                "or use __call__ which handles initialization."
            )
        # Delegate to the module-level jitted core (phase-free sin/cos estimator) so
        # the current weights are always used: no stale-weight baking via a static self.
        return _rff_feature_map(X, self._W)

    def __call__(self, X: Float[Array, "n d"], Y: Float[Array, "m d"]) -> Float[Array, "n m"]:
        """
        Approximate kernel matrix via random features.

        K(X, Y) ≈ φ(X) @ φ(Y).T

        Parameters:
            X: First set of points, shape (n, d)
            Y: Second set of points, shape (m, d)

        Returns:
            Approximate kernel matrix of shape (n, m)
        """
        # Initialize weights if needed
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have same number of features")

        self._initialize_weights(X.shape[1])

        # Compute feature maps
        phi_X = self.feature_map(X)
        phi_Y = self.feature_map(Y)

        # Kernel matrix is dot product of feature maps
        return jnp.dot(phi_X, phi_Y.T)

    def self_similarity(self, X: Float[Array, "n d"]) -> Float[Array, "n n"]:
        """
        Compute K(X, X) efficiently.

        Parameters:
            X: Input points, shape (n, d)

        Returns:
            Self-similarity matrix of shape (n, n)
        """
        self._initialize_weights(X.shape[1])
        phi_X = self.feature_map(X)
        return jnp.dot(phi_X, phi_X.T)
