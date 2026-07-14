"""High-level API for KLRfome."""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal
import jax.numpy as jnp
import numpy as np
import warnings

from .data.formats import TrainingData, RasterStack, SampleCollection
from .kernels.rbf import RBFKernel
from .kernels.rff import RandomFourierFeatures
from .kernels.distribution import MeanEmbeddingKernel
from .kernels.wasserstein import (
    WassersteinKernel,
    SlicedWassersteinDistance,
    estimate_sigma_from_distances,
)
from .models.klr import KernelLogisticRegression, KLRFitResult
from .models.distribution import DistributionClassifier
from .models.spec import ModelSpec


@dataclass
class KLRfome:
    """
    High-level interface for KLRfome modeling.

    Example usage:

        # Initialize model with mean embedding kernel (default)
        model = KLRfome(
            sigma=0.5,
            lambda_reg=0.1,
            n_rff_features=256,
            window_size=5
        )

        # Or use Wasserstein kernel for shape-aware distribution comparison
        model = KLRfome(
            sigma=0.5,
            lambda_reg=0.1,
            kernel_type='wasserstein',
            n_projections=100,
            window_size=5
        )

        # Prepare training data
        training_data = model.prepare_data(
            raster_stack=my_rasters,
            sites=site_geodataframe,
            n_background=1000,
            samples_per_location=20
        )

        # Fit model
        model.fit(training_data)

        # Predict
        prediction_raster = model.predict(prediction_rasters)

    Parameters:
        sigma: Kernel bandwidth (controls similarity decay)
        lambda_reg: KLR regularization strength
        kernel_type: 'mean_embedding' (default) or 'wasserstein'
        n_rff_features: Number of random Fourier features (only for mean_embedding, 0 for exact)
        n_projections: Number of random projections (only for wasserstein)
        wasserstein_p: Order of Wasserstein distance, 1 or 2 (only for wasserstein)
        bucket_width: For Wasserstein kernel - group distributions by size ranges (e.g., 25 for [0-24], [25-49], ...)
                     Default None uses exact mode. Recommended: 25 for real data with variable sizes.
        bucket_ceil: For Wasserstein kernel - resample to bucket ceiling (max) vs median
        bucket_cap: For Wasserstein kernel - global maximum sample size (e.g., 2500)
        window_size: Focal window size for prediction
        seed: Random seed for reproducibility
    """

    sigma: float = 0.5  # Kernel bandwidth (0.5 works well for scaled data)
    lambda_reg: float = 0.1
    kernel_type: Literal["mean_embedding", "wasserstein"] = "mean_embedding"
    n_rff_features: int = 256  # Use RFF approximation by default (faster)
    n_projections: int = 100  # For Wasserstein kernel
    wasserstein_p: Literal[1, 2] = 2  # Order of Wasserstein distance
    n_quantiles: int = (
        128  # For Wasserstein: single global-Q quantile representation (fit AND predict)
    )
    bucket_width: Optional[int] = (
        None  # For Wasserstein bucketing (legacy; superseded by n_quantiles)
    )
    bucket_ceil: bool = True  # For Wasserstein bucketing
    bucket_cap: Optional[int] = None  # For Wasserstein bucketing
    window_size: int = 3
    seed: int = 42
    scale_features: bool = True  # z-score covariates (matches R format_site_data)
    auto_sigma: bool = True  # calibrate sigma to the data at fit() time (median-distance heuristic)
    embedding_kernel: Literal["linear", "rbf"] = "linear"
    spec: Optional[ModelSpec] = None

    # Fitted attributes (set after fit())
    _training_data: Optional[TrainingData] = field(default=None, init=False)
    _feature_means: Optional[jnp.ndarray] = field(default=None, init=False)
    _feature_stds: Optional[jnp.ndarray] = field(default=None, init=False)
    _similarity_matrix: Optional[jnp.ndarray] = field(default=None, init=False)
    _fit_result: Optional[KLRFitResult] = field(default=None, init=False)
    _distribution_kernel: Optional[Union[MeanEmbeddingKernel, WassersteinKernel]] = field(
        default=None, init=False
    )
    _klr: Optional[KernelLogisticRegression] = field(default=None, init=False)
    _core_model: Optional[DistributionClassifier] = field(default=None, init=False)
    _resolved_spec: ModelSpec = field(init=False)

    def __post_init__(self):
        """Initialize kernel and KLR model."""
        self._resolved_spec = self.spec or ModelSpec.from_legacy(
            kernel_type=self.kernel_type,
            n_rff_features=self.n_rff_features,
            wasserstein_p=self.wasserstein_p,
            n_projections=self.n_projections,
            n_quantiles=self.n_quantiles,
            embedding_kernel=self.embedding_kernel,
        )
        # Initialize kernel based on type
        if self.kernel_type == "wasserstein":
            self._distribution_kernel = WassersteinKernel(
                sigma=self.sigma,
                n_projections=self.n_projections,
                p=self.wasserstein_p,
                seed=self.seed,
            )
        else:
            # Mean embedding kernel (default)
            if self.n_rff_features > 0:
                base_kernel = RandomFourierFeatures(
                    sigma=self.sigma, n_features=self.n_rff_features, seed=self.seed
                )
            else:
                base_kernel = RBFKernel(sigma=self.sigma)
            self._distribution_kernel = MeanEmbeddingKernel(base_kernel)

        self._klr = KernelLogisticRegression(lambda_reg=self.lambda_reg)

    def prepare_data(
        self,
        raster_stack: Union[RasterStack, List[str]],
        sites: "geopandas.GeoDataFrame",  # type: ignore
        n_background: int = 1000,
        samples_per_location: int = 20,
        background_exclusion_buffer: Optional[float] = None,
        site_buffer: Optional[float] = None,
    ) -> TrainingData:
        """
        Prepare training data from rasters and site locations.

        Parameters:
            raster_stack: RasterStack object or list of raster file paths
            sites: GeoDataFrame with site geometries
            n_background: Number of background sample locations
            samples_per_location: Samples to extract per site/background
            background_exclusion_buffer: Buffer around sites to exclude from background
            site_buffer: Buffer to apply around site points (if points, not polygons)

        Returns:
            TrainingData object ready for fitting
        """
        from klrfome.io.vector import (
            extract_distribution_at_points,
            generate_background_points,
        )

        if isinstance(raster_stack, list):
            raster_stack = RasterStack.from_files(raster_stack)

        # Extract site samples as genuine multi-pixel distributions (neighbouring
        # cells), not a single repeated pixel. site_buffer is retained for API
        # compatibility but no longer needed to avoid degenerate bags.
        site_collections = extract_distribution_at_points(
            raster_stack,
            sites,
            n_samples=samples_per_location,
            label=1,
            random_seed=self.seed,
        )
        for index, bag in enumerate(site_collections):
            bag.id = f"site-{index:05d}"
            bag.group_id = bag.id
            bag.metadata = {**(bag.metadata or {}), "source": "site"}

        # Generate exclusion geometries if buffer specified
        exclusion_geoms = None
        if background_exclusion_buffer is not None:
            exclusion_geoms = [site.buffer(background_exclusion_buffer) for site in sites.geometry]

        # Generate background points
        background_points = generate_background_points(
            raster_stack,
            exclusion_geoms=exclusion_geoms,
            n_points=n_background,
            random_seed=self.seed,
        )

        # Extract background samples as genuine multi-pixel distributions.
        background_collections = extract_distribution_at_points(
            raster_stack,
            background_points,
            n_samples=samples_per_location,
            label=0,
            random_seed=self.seed,
        )
        for index, bag in enumerate(background_collections):
            bag.id = f"background-{index:05d}"
            bag.group_id = bag.id
            bag.metadata = {**(bag.metadata or {}), "source": "background"}

        collections = site_collections + background_collections

        return TrainingData(
            collections=collections, feature_names=raster_stack.band_names, crs=raster_stack.crs
        )

    def fit(self, training_data: TrainingData) -> "KLRfome":
        """
        Fit the KLRfome model.

        Parameters:
            training_data: Prepared TrainingData object

        Returns:
            self (for method chaining)
        """
        self._core_model = DistributionClassifier(
            spec=self._resolved_spec,
            sigma=self.sigma,
            lambda_reg=self.lambda_reg,
            scale_features=self.scale_features,
            auto_sigma=self.auto_sigma,
            seed=self.seed,
        ).fit(training_data)
        self._training_data = self._core_model.training_data_
        self._similarity_matrix = self._core_model.gram_matrix_
        self._fit_result = self._core_model.fit_result_  # type: ignore[assignment]
        if self._core_model.preprocessor_ is not None:
            self._feature_means = jnp.asarray(self._core_model.preprocessor_.means)
            self._feature_stds = jnp.asarray(self._core_model.preprocessor_.scales)
        if not self._fit_result.converged:
            warnings.warn(
                f"{self._resolved_spec.method_id} fitting did not converge: "
                f"{self._fit_result.failure_reason}"
            )
        return self

    def predict(
        self,
        raster_stack: Union[RasterStack, List[str]],
        batch_size: int = 1000,
        show_progress: bool = True,
    ) -> jnp.ndarray:
        """
        Generate predictions across a raster extent.

        Parameters:
            raster_stack: Rasters to predict on (must have same bands as training)
            batch_size: Windows to process per batch
            show_progress: Show progress bar

        Returns:
            2D array of predicted probabilities
        """
        if self._fit_result is None or self._core_model is None:
            raise RuntimeError("Model must be fit before prediction")
        if self._training_data is None:
            raise RuntimeError("Training data not available")
        if isinstance(raster_stack, list):
            raster_stack = RasterStack.from_files(raster_stack)
        if raster_stack.band_names != self._training_data.feature_names:
            raise ValueError("Prediction raster band order differs from training features")

        from tqdm import tqdm

        pad = self.window_size // 2
        data = np.asarray(raster_stack.data)
        padded = np.pad(data, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        coordinates = [
            (row, col) for row in range(raster_stack.height) for col in range(raster_stack.width)
        ]
        output = np.full(len(coordinates), np.nan, dtype=float)
        iterator = range(0, len(coordinates), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")
        for start in iterator:
            chunk = coordinates[start : start + batch_size]
            bags = []
            output_indices = []
            for local_index, (row, col) in enumerate(chunk):
                window = padded[:, row : row + self.window_size, col : col + self.window_size]
                samples = window.reshape(window.shape[0], -1).T
                valid = np.isfinite(samples).all(axis=1)
                if raster_stack.nodata is not None:
                    valid &= (samples != raster_stack.nodata).all(axis=1)
                samples = samples[valid]
                if len(samples) == 0:
                    continue
                bags.append(SampleCollection(samples, 0, f"prediction-{start + local_index}"))
                output_indices.append(start + local_index)
            if bags:
                prediction_data = TrainingData(
                    bags,
                    list(raster_stack.band_names),
                    crs=raster_stack.crs,
                    study_design=self._training_data.study_design,
                )
                output[output_indices] = np.asarray(self._core_model.predict_bags(prediction_data))
        return jnp.asarray(output.reshape(raster_stack.height, raster_stack.width))

    def save_prediction(
        self, path: str, prediction: jnp.ndarray, reference_raster: Optional[RasterStack] = None
    ):
        """
        Save prediction array as a GeoTIFF.

        Parameters:
            path: Output file path
            prediction: Prediction array (2D)
            reference_raster: Raster to use for georeferencing (if None, uses training data raster if available)
        """
        if reference_raster is None:
            # Try to get from training data if available
            # For now, raise error - user must provide reference
            raise ValueError(
                "reference_raster must be provided. " "Use the raster stack used for prediction."
            )

        # Create a temporary RasterStack with prediction data
        # We'll use the reference raster's metadata
        temp_stack = RasterStack(
            data=jnp.expand_dims(prediction, axis=0),  # Add band dimension
            transform=reference_raster.transform,
            crs=reference_raster.crs,
            band_names=["prediction"],
            nodata=None,
        )

        temp_stack.save(path, data=prediction)

    def cross_validate(
        self, training_data: TrainingData, n_folds: int = 5, stratified: bool = True
    ) -> dict:
        """
        Perform k-fold cross-validation.

        Parameters:
            training_data: Training data
            n_folds: Number of folds
            stratified: Whether to stratify by label

        Returns:
            Dictionary with metrics per fold and aggregated statistics
        """
        from klrfome.utils.validation import cross_validate

        return cross_validate(self, training_data, n_folds, stratified, self.seed)

    def _generate_background_points(self, *args, **kwargs):
        """Generate random background sample locations."""
        # Implementation is now in io.vector.generate_background_points
        from klrfome.io.vector import generate_background_points

        return generate_background_points(*args, **kwargs)

    def estimate_sigma(
        self, training_data: Optional[TrainingData] = None, percentile: float = 50.0
    ) -> float:
        """
        Estimate a reasonable sigma based on pairwise distances.

        Should be called before fit() to help choose sigma.
        Uses the median distance heuristic.

        Parameters:
            training_data: Data to estimate from (uses fitted data if None)
            percentile: Which percentile of distances to use (default: median)

        Returns:
            Suggested sigma value
        """
        if training_data is None:
            training_data = self._training_data
        if training_data is None:
            raise ValueError("No training data available")

        if self.kernel_type == "wasserstein":
            sw = SlicedWassersteinDistance(
                n_projections=self.n_projections, p=self.wasserstein_p, seed=self.seed
            )
            distances = sw.pairwise_distances(training_data.collections)
            return estimate_sigma_from_distances(distances, percentile)
        else:
            # For mean embeddings, estimate based on feature-space distances
            # Compute mean of each collection and measure pairwise Euclidean
            import numpy as np

            means = []
            for coll in training_data.collections:
                means.append(np.mean(np.array(coll.samples), axis=0))
            means = np.stack(means)

            # Pairwise Euclidean distances
            n = len(means)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(means[i] - means[j])
                    distances[i, j] = d
                    distances[j, i] = d

            return float(estimate_sigma_from_distances(jnp.array(distances), percentile))
