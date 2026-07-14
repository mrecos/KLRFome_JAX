"""Focal window prediction for KLRfome."""

import jax.numpy as jnp
from jax import jit, vmap
import jax.lax as lax
from functools import partial
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm
from jaxtyping import Array, Float

from ..kernels.distribution import MeanEmbeddingKernel
from ..kernels.wasserstein import WassersteinKernel, _resample_sorted_projections
from ..kernels.rff import _rff_feature_map
from ..data.formats import TrainingData, RasterStack


@dataclass
class FocalPredictor:
    """
    Focal window prediction for KLRfome.

    Slides a window across the raster stack, computing similarity
    between each window and training data, then applying the fitted
    KLR model.

    Parameters:
        distribution_kernel: MeanEmbeddingKernel instance
        klr_alpha: Fitted coefficients from KLR
        training_data: Original training TrainingData
        window_size: Size of focal window (e.g., 3 for 3x3)
        use_gpu: Whether to use GPU acceleration (for future use)
    """

    distribution_kernel: MeanEmbeddingKernel
    klr_alpha: Float[Array, "n_train"]
    training_data: TrainingData
    window_size: int = 3
    use_gpu: bool = True

    def __post_init__(self):
        """Precompute training mean embeddings for efficiency."""
        if self.distribution_kernel._use_rff:
            self._training_embeddings = self._compute_training_embeddings()
            # Extract RFF weights for JIT compatibility
            base_kernel = self.distribution_kernel.base_kernel
            if hasattr(base_kernel, "_W") and base_kernel._W is not None:
                self._rff_W = base_kernel._W
                self._rff_b = base_kernel._b
                self._rff_n_features = base_kernel.n_features
            else:
                self._rff_W = None
                self._rff_b = None
                self._rff_n_features = None
        else:
            self._training_embeddings = None
            self._rff_W = None
            self._rff_b = None
            self._rff_n_features = None
            # Exact mean-embedding path: pack the (variable-length) training bags
            # into a padded (n_train, max_m, d) tensor plus a validity mask so the
            # kernel for every pixel can be computed in a single jit+vmap pass
            # instead of a Python loop with a host-sync per training bag.
            bags = [jnp.asarray(c.samples) for c in self.training_data.collections]
            n_train = len(bags)
            max_m = max(int(b.shape[0]) for b in bags)
            d = int(bags[0].shape[1])
            stack = jnp.zeros((n_train, max_m, d))
            mask = jnp.zeros((n_train, max_m))
            counts = []
            for t, b in enumerate(bags):
                mt = int(b.shape[0])
                stack = stack.at[t, :mt, :].set(b)
                mask = mask.at[t, :mt].set(1.0)
                counts.append(mt)
            self._train_stack = stack
            self._train_mask = mask
            self._train_count = jnp.asarray(counts, dtype=jnp.float32)
            self._exact_sigma = float(self.distribution_kernel.base_kernel.sigma)

    def _compute_training_embeddings(self) -> Float[Array, "n_train D"]:
        """Precompute mean embeddings for all training collections."""
        embeddings = []
        base_kernel = self.distribution_kernel.base_kernel

        # Initialize weights if needed (use first collection)
        if hasattr(base_kernel, "_initialize_weights"):
            first_coll = self.training_data.collections[0]
            base_kernel._initialize_weights(first_coll.n_features)

        for coll in self.training_data.collections:
            phi = base_kernel.feature_map(coll.samples)
            embeddings.append(jnp.mean(phi, axis=0))

        return jnp.stack(embeddings)

    def predict_window(self, window_samples: Float[Array, "m d"]) -> float:
        """
        Predict probability for a single focal window.

        Parameters:
            window_samples: Samples from the focal window, shape (m, n_features)
                           where m = window_size * window_size (excluding nodata)

        Returns:
            Predicted probability of "site" class
        """
        if self._training_embeddings is not None:
            # RFF path: compute mean embedding and dot with training embeddings
            base_kernel = self.distribution_kernel.base_kernel
            phi = base_kernel.feature_map(window_samples)
            window_embedding = jnp.mean(phi, axis=0)
            K_new = jnp.dot(window_embedding, self._training_embeddings.T)
        else:
            # Exact path: compute kernel with each training collection
            # Round to 3 decimals to match R's KLR_predict (line 132)
            K_new = jnp.array(
                [
                    round(float(self.distribution_kernel(window_samples, coll.samples)), 3)
                    for coll in self.training_data.collections
                ]
            )

        # Apply KLR prediction: η = K_new @ alpha, then sigmoid
        eta = jnp.dot(K_new, self.klr_alpha)
        return float(1 / (1 + jnp.exp(-eta)))

    def predict_raster(
        self, raster_stack: RasterStack, batch_size: int = 1000, show_progress: bool = True
    ) -> Float[Array, "height width"]:
        """
        Predict across entire raster using focal windows.

        Parameters:
            raster_stack: Input raster stack
            batch_size: Number of windows to process in parallel
            show_progress: Whether to show progress bar

        Returns:
            Prediction raster with same spatial extent as input
        """
        height, width = raster_stack.height, raster_stack.width
        pad = self.window_size // 2

        # Pad raster to handle edges
        padded_data = jnp.pad(raster_stack.data, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")

        # Generate all window center coordinates
        rows, cols = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
        coords = jnp.stack([rows.ravel(), cols.ravel()], axis=1)

        # Batch prediction
        n_pixels = coords.shape[0]
        predictions = []

        iterator = range(0, n_pixels, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")

        # Check if using exact kernel (non-JIT path needed)
        use_exact_kernel = not self.distribution_kernel._use_rff

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n_pixels)
            batch_coords = coords[start_idx:end_idx]

            if use_exact_kernel:
                # Exact kernel: fully JIT+vmap'd over the padded/masked training
                # tensor (fast AND exact, replacing the old per-pixel Python loop).
                batch_preds = self._predict_batch_exact_jit(
                    padded_data,
                    batch_coords,
                    pad,
                    self.window_size,
                    self._train_stack,
                    self._train_mask,
                    self._train_count,
                    self.klr_alpha,
                    self._exact_sigma,
                    True,  # round kernel to 3 decimals to match R's KLR_predict
                )
            else:
                # RFF path: use JIT-compiled batch
                batch_preds = self._predict_batch(
                    padded_data,
                    batch_coords,
                    pad,
                    self.window_size,
                    self._training_embeddings,
                    self.klr_alpha,
                    self.distribution_kernel._use_rff,
                    self._rff_W,
                    self._rff_b,
                    self._rff_n_features,
                )
            predictions.append(batch_preds)

        predictions = jnp.concatenate(predictions)
        return predictions.reshape(height, width)

    @staticmethod
    @partial(jit, static_argnames=["window_size", "use_rff", "rff_n_features"])
    def _predict_batch(
        padded_data: Float[Array, "bands h w"],
        coords: Float[Array, "batch 2"],
        pad: int,
        window_size: int,
        training_embeddings: Optional[Float[Array, "n_train D"]],
        klr_alpha: Float[Array, "n_train"],
        use_rff: bool,
        rff_W: Optional[Float[Array, "d D"]],
        rff_b: Optional[Float[Array, "D"]],
        rff_n_features: Optional[int],
    ) -> Float[Array, "batch"]:
        """
        JIT-compiled batch prediction.

        Parameters:
            padded_data: Padded raster data
            coords: Batch of (row, col) coordinates
            pad: Padding size
            window_size: Size of focal window
            training_embeddings: Precomputed training embeddings (for RFF path)
            klr_alpha: Fitted KLR coefficients
            use_rff: Whether using RFF approximation
            rff_W: RFF weight matrix (for RFF path)
            rff_b: RFF bias vector (for RFF path)
            rff_n_features: Number of RFF features (for RFF path)

        Returns:
            Batch of predictions
        """

        def predict_single(coord):
            r, c = coord
            # Extract the window centred on output pixel (r, c). In padded
            # coordinates the slice starts at (r, c) and spans window_size, which
            # is centred on (r+pad, c+pad) == original (r, c).
            window = lax.dynamic_slice(
                padded_data, (0, r, c), (padded_data.shape[0], window_size, window_size)
            )
            # Reshape to (n_samples, n_features)
            # window is (bands, window_size, window_size)
            window_samples = window.reshape(window.shape[0], -1).T

            # Compute prediction
            if use_rff and training_embeddings is not None and rff_W is not None:
                # RFF path: phase-free sin/cos features, identical to
                # RandomFourierFeatures.feature_map (rff_b/rff_n_features now unused).
                phi = _rff_feature_map(window_samples, rff_W)
                window_embedding = jnp.mean(phi, axis=0)
                K_new = jnp.dot(window_embedding, training_embeddings.T)
            else:
                # Exact path - not supported in JIT
                # Return zeros as placeholder (exact kernels should not use JIT path)
                K_new = jnp.zeros(klr_alpha.shape[0])

            # Apply KLR
            eta = jnp.dot(K_new, klr_alpha)
            return 1 / (1 + jnp.exp(-eta))

        return vmap(predict_single)(coords)

    def _predict_batch_exact(
        self,
        padded_data: Float[Array, "bands h w"],
        coords: Float[Array, "batch 2"],
        pad: int,
        window_size: int,
        distribution_kernel: MeanEmbeddingKernel,
        training_data: TrainingData,
        klr_alpha: Float[Array, "n_train"],
    ) -> Float[Array, "batch"]:
        """
        Non-JIT batch prediction for exact kernel path.

        This is slower than RFF but provides exact kernel computation.
        """
        predictions = []

        for coord in coords:
            r, c = int(coord[0]), int(coord[1])
            # Extract window (accounting for padding offset)
            window = padded_data[:, r : r + window_size, c : c + window_size]
            # Reshape to (n_samples, n_features)
            window_samples = window.reshape(window.shape[0], -1).T

            # Compute kernel with each training collection
            K_new = jnp.array(
                [
                    jnp.round(distribution_kernel(window_samples, coll.samples), 3)
                    for coll in training_data.collections
                ]
            )

            # Apply KLR
            eta = jnp.dot(K_new, klr_alpha)
            pred = 1 / (1 + jnp.exp(-eta))
            predictions.append(pred)

        return jnp.array(predictions)

    @staticmethod
    @partial(jit, static_argnames=["window_size", "round_kernel"])
    def _predict_batch_exact_jit(
        padded_data: Float[Array, "bands h w"],
        coords: Float[Array, "batch 2"],
        pad: int,
        window_size: int,
        train_stack: Float[Array, "n_train max_m d"],
        train_mask: Float[Array, "n_train max_m"],
        train_count: Float[Array, "n_train"],
        klr_alpha: Float[Array, "n_train"],
        sigma: float,
        round_kernel: bool,
    ) -> Float[Array, "batch"]:
        """
        Vectorized exact mean-embedding focal prediction.

        For each focal window W (window_size**2 samples) and each training bag B_t
        (padded to max_m, masked), computes the mean-embedding RBF similarity

            K(W, B_t) = (1 / (|W| |B_t|)) * sum_{i,j} exp(-||w_i - b_{t,j}||^2 / 2 sigma^2)

        using the identity ||w - b||^2 = ||w||^2 + ||b||^2 - 2 w.b, then applies the
        fitted KLR coefficients. Fully JIT-compiled and vmapped over pixels, so the
        accurate path runs at RFF-like speed.
        """
        n_bands = padded_data.shape[0]
        inv_2s2 = 1.0 / (2.0 * sigma**2)
        m = window_size * window_size  # static window sample count
        b_sq = jnp.sum(train_stack**2, axis=2)  # (n_train, max_m)

        def predict_single(coord):
            r, c = coord[0], coord[1]
            window = lax.dynamic_slice(padded_data, (0, r, c), (n_bands, window_size, window_size))
            w = window.reshape(n_bands, -1).T  # (m, d)
            w_sq = jnp.sum(w**2, axis=1)  # (m,)
            cross = jnp.einsum("nmd,id->nmi", train_stack, w)  # (n_train, max_m, m)
            sq_dist = b_sq[:, :, None] + w_sq[None, None, :] - 2.0 * cross
            rbf = jnp.exp(-sq_dist * inv_2s2) * train_mask[:, :, None]
            # mean over valid training samples (count) and over window samples (m)
            K = jnp.sum(rbf, axis=(1, 2)) / (train_count * m)  # (n_train,)
            if round_kernel:
                K = jnp.round(K, 3)
            eta = jnp.dot(K, klr_alpha)
            return 1.0 / (1.0 + jnp.exp(-eta))

        return vmap(predict_single)(coords)


@dataclass
class WassersteinFocalPredictor:
    """
    Focal window prediction using Wasserstein kernel.

    Pre-computes projected and sorted training data for efficient
    repeated comparisons during focal window prediction.

    OPTIMIZED: Uses JIT compilation and vectorization for speed.

    Parameters:
        wasserstein_kernel: Configured WassersteinKernel instance
        training_collections: Training data SampleCollections
        klr_alpha: Fitted KLR coefficients
        window_size: Size of focal window (e.g., 5 for 5x5)
    """

    wasserstein_kernel: WassersteinKernel
    training_collections: list  # List[SampleCollection]
    klr_alpha: Float[Array, "n_train"]
    window_size: int = 5
    n_quantiles: int = 128

    def __post_init__(self):
        """Pre-compute the Q-quantile training representation."""
        sw = self.wasserstein_kernel._sw
        dim = self.training_collections[0].samples.shape[1]
        sw._ensure_projections(dim)

        # Store projections for use during prediction
        self._projections = sw._projections
        self._p = sw.p
        self._sigma = self.wasserstein_kernel.sigma
        self._n_projections = sw.n_projections

        # Represent every training bag by the SAME number of sorted quantile points
        # (single global Q). This matches how the Wasserstein kernel is fit
        # (build_similarity_matrix(n_quantiles=...)), so fit and predict use one
        # consistent representation even when bags have different sizes -- avoiding the
        # pair-dependent interpolation that made raw native-size SW inconsistent.
        self._training_stacked = sw.quantile_embeddings(
            self.training_collections, self.n_quantiles
        )  # (n_train, Q, L)
        self._uniform_samples = True
        self._n_samples_per_collection = self.n_quantiles
        self._training_projected_sorted = None

    def predict_window(self, window_samples: Float[Array, "m d"]) -> float:
        """
        Predict probability for a single focal window.

        Parameters:
            window_samples: Feature vectors from the focal window

        Returns:
            Predicted probability of site presence
        """
        # Project, sort, and resample the window to the same Q quantile points the
        # training bags use, so the comparison happens in one consistent space.
        window_proj = window_samples @ self._projections.T  # (m, L)
        window_sorted = jnp.sort(window_proj, axis=0)
        window_q = _resample_sorted_projections(window_sorted, self.n_quantiles)  # (Q, L)

        similarities = self._compute_similarities_vectorized(window_q)

        # Apply KLR prediction
        eta = jnp.dot(similarities, self.klr_alpha)
        return float(1.0 / (1.0 + jnp.exp(-eta)))

    def _compute_similarities_vectorized(
        self, window_sorted: Float[Array, "m L"]
    ) -> Float[Array, "n_train"]:
        """
        VECTORIZED similarity computation for uniform sample sizes.

        Computes all training similarities in one operation.
        """
        # window_sorted: (m, L)
        # training_stacked: (n_train, m, L)
        if self._p == 2:
            # SW_2: sqrt of mean squared difference
            # (n_train, m, L) - (m, L) -> broadcast to (n_train, m, L)
            diff = self._training_stacked - window_sorted[None, :, :]
            sw_sq = jnp.mean(diff**2, axis=(1, 2))  # (n_train,)
            sw_dist = jnp.sqrt(sw_sq)
        else:
            # SW_1: mean absolute difference
            diff = self._training_stacked - window_sorted[None, :, :]
            sw_dist = jnp.mean(jnp.abs(diff), axis=(1, 2))  # (n_train,)

        # RBF kernel on distance
        similarities = jnp.exp(-(sw_dist**2) / (2 * self._sigma**2))
        return similarities

    def _compute_similarities_loop(
        self, window_sorted: Float[Array, "m L"]
    ) -> Float[Array, "n_train"]:
        """Fallback loop-based similarity for non-uniform sample sizes."""
        similarities = []

        for train_sorted in self._training_projected_sorted:
            if self._p == 2:
                if window_sorted.shape[0] == train_sorted.shape[0]:
                    sw_sq = jnp.mean((window_sorted - train_sorted) ** 2)
                    sw_dist = jnp.sqrt(sw_sq)
                else:
                    # Interpolation for unequal sizes
                    from ..kernels.wasserstein import wasserstein_1d_p2

                    def w2_sq(w, t):
                        return wasserstein_1d_p2(w, t) ** 2

                    sw_sq = jnp.mean(vmap(w2_sq)(window_sorted.T, train_sorted.T))
                    sw_dist = jnp.sqrt(sw_sq)
            else:
                if window_sorted.shape[0] == train_sorted.shape[0]:
                    sw_dist = jnp.mean(jnp.abs(window_sorted - train_sorted))
                else:
                    from ..kernels.wasserstein import wasserstein_1d_p1

                    sw_dist = jnp.mean(vmap(wasserstein_1d_p1)(window_sorted.T, train_sorted.T))

            sim = jnp.exp(-(sw_dist**2) / (2 * self._sigma**2))
            similarities.append(sim)

        return jnp.array(similarities)

    def predict_raster(
        self, raster_stack: RasterStack, batch_size: int = 1000, show_progress: bool = True
    ) -> Float[Array, "height width"]:
        """
        Predict across entire raster using focal windows.

        OPTIMIZED: Uses JIT-compiled batch processing.
        """
        height, width = raster_stack.height, raster_stack.width
        pad = self.window_size // 2

        # Pad raster to handle edges
        padded_data = jnp.pad(raster_stack.data, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")

        # Generate all window center coordinates
        rows, cols = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
        coords = jnp.stack([rows.ravel(), cols.ravel()], axis=1)

        # Batch prediction
        n_pixels = coords.shape[0]
        predictions = []

        iterator = range(0, n_pixels, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting (Wasserstein)")

        # Use JIT-compiled path for uniform samples
        if self._uniform_samples and self._training_stacked is not None:
            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, n_pixels)
                batch_coords = coords[start_idx:end_idx]

                batch_preds = self._predict_batch_jit(
                    padded_data,
                    batch_coords,
                    pad,
                    self.window_size,
                    self._projections,
                    self._training_stacked,
                    self.klr_alpha,
                    self._sigma,
                    self._p,
                    self._n_samples_per_collection,
                )
                predictions.append(batch_preds)
        else:
            # Fallback for non-uniform
            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, n_pixels)
                batch_coords = coords[start_idx:end_idx]

                batch_preds = self._predict_batch_loop(padded_data, batch_coords, pad)
                predictions.append(batch_preds)

        predictions = jnp.concatenate(predictions)
        return predictions.reshape(height, width)

    @staticmethod
    @partial(jit, static_argnames=["window_size", "p", "n_train_samples"])
    def _predict_batch_jit(
        padded_data: Float[Array, "bands h w"],
        coords: Float[Array, "batch 2"],
        pad: int,
        window_size: int,
        projections: Float[Array, "L d"],
        training_stacked: Float[Array, "n_train m L"],
        klr_alpha: Float[Array, "n_train"],
        sigma: float,
        p: int,
        n_train_samples: int,
    ) -> Float[Array, "batch"]:
        """
        JIT-compiled batch prediction for Wasserstein kernel.

        FAST: Fully vectorized, no Python loops.
        Uses linear interpolation when window and training have different sample sizes.
        """
        n_bands = padded_data.shape[0]
        n_window_samples = window_size * window_size

        def predict_single(coord):
            r, c = coord[0], coord[1]
            # Extract window using dynamic_slice for JIT compatibility
            window = lax.dynamic_slice(padded_data, (0, r, c), (n_bands, window_size, window_size))
            # Reshape: (bands, ws, ws) -> (ws*ws, bands)
            window_samples = window.reshape(n_bands, -1).T

            # Project and sort
            window_proj = window_samples @ projections.T  # (n_window, L)
            window_sorted = jnp.sort(window_proj, axis=0)

            # If sizes match, use direct comparison
            # If not, resample window to match training size using linear interpolation
            def wasserstein_distance():
                # Resample window to match training size
                # Use quantile matching: evaluate window CDF at training quantiles
                n_w = n_window_samples
                n_t = n_train_samples

                if n_w == n_t:
                    # Direct comparison
                    diff = training_stacked - window_sorted[None, :, :]
                else:
                    # Linear interpolation: resample window to training size
                    # Quantiles for training points
                    t_quantiles = (jnp.arange(n_t) + 0.5) / n_t  # (n_t,)
                    w_quantiles = (jnp.arange(n_w) + 0.5) / n_w  # (n_w,)

                    # Interpolate window values at training quantiles
                    # For each projection dimension, interpolate
                    def interp_projection(w_sorted_col):
                        # w_sorted_col: (n_w,)
                        # Interpolate to get values at t_quantiles
                        return jnp.interp(t_quantiles, w_quantiles, w_sorted_col)

                    # Apply to all projections: window_sorted is (n_w, L)
                    window_resampled = vmap(interp_projection, in_axes=1, out_axes=1)(
                        window_sorted
                    )  # (n_t, L)

                    # Now compare
                    diff = training_stacked - window_resampled[None, :, :]

                if p == 2:
                    sw_sq = jnp.mean(diff**2, axis=(1, 2))
                    sw_dist = jnp.sqrt(sw_sq)
                else:
                    sw_dist = jnp.mean(jnp.abs(diff), axis=(1, 2))

                return sw_dist

            sw_dist = wasserstein_distance()

            # RBF kernel
            similarities = jnp.exp(-(sw_dist**2) / (2 * sigma**2))

            # KLR prediction
            eta = jnp.dot(similarities, klr_alpha)
            return 1.0 / (1.0 + jnp.exp(-eta))

        return vmap(predict_single)(coords)

    def _predict_batch_loop(
        self, padded_data: Float[Array, "bands h w"], coords: Float[Array, "batch 2"], pad: int
    ) -> Float[Array, "batch"]:
        """
        Fallback batch prediction for non-uniform sample sizes.

        SLOW: Uses Python loops, avoid if possible.
        """
        predictions = []
        window_size = self.window_size

        for coord in coords:
            r, c = int(coord[0]), int(coord[1])
            window = padded_data[:, r : r + window_size, c : c + window_size]
            window_samples = window.reshape(window.shape[0], -1).T

            window_proj = window_samples @ self._projections.T
            window_sorted = jnp.sort(window_proj, axis=0)

            similarities = self._compute_similarities_loop(window_sorted)

            eta = jnp.dot(similarities, self.klr_alpha)
            pred = 1.0 / (1.0 + jnp.exp(-eta))
            predictions.append(float(pred))

        return jnp.array(predictions)
