"""High-level API for KLRfome."""

from dataclasses import dataclass, field
from typing import Optional, Union, List
import jax.numpy as jnp
import warnings

from .data.formats import TrainingData, RasterStack, SampleCollection
from .kernels.rbf import RBFKernel
from .kernels.rff import RandomFourierFeatures
from .kernels.distribution import MeanEmbeddingKernel
from .models.klr import KernelLogisticRegression, KLRFitResult
from .prediction.focal import FocalPredictor


@dataclass
class KLRfome:
    """
    High-level interface for KLRfome modeling.
    
    Example usage:
    
        # Initialize model
        model = KLRfome(
            sigma=1.0,
            lambda_reg=0.1,
            n_rff_features=256,
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
        
        # Save
        model.save_prediction("output.tif", prediction_raster)
    
    Parameters:
        sigma: RBF kernel bandwidth
        lambda_reg: KLR regularization strength
        n_rff_features: Number of random Fourier features (0 for exact kernel)
        window_size: Focal window size for prediction
        seed: Random seed for reproducibility
    """
    sigma: float = 1.0
    lambda_reg: float = 0.1
    n_rff_features: int = 256
    window_size: int = 3
    seed: int = 42
    
    # Fitted attributes (set after fit())
    _training_data: Optional[TrainingData] = field(default=None, init=False)
    _similarity_matrix: Optional[jnp.ndarray] = field(default=None, init=False)
    _fit_result: Optional[KLRFitResult] = field(default=None, init=False)
    _distribution_kernel: Optional[MeanEmbeddingKernel] = field(default=None, init=False)
    _klr: Optional[KernelLogisticRegression] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize kernel and KLR model."""
        # Initialize kernel
        if self.n_rff_features > 0:
            base_kernel = RandomFourierFeatures(
                sigma=self.sigma,
                n_features=self.n_rff_features,
                seed=self.seed
            )
        else:
            base_kernel = RBFKernel(sigma=self.sigma)
        
        self._distribution_kernel = MeanEmbeddingKernel(base_kernel)
        self._klr = KernelLogisticRegression(lambda_reg=self.lambda_reg)
    
    def prepare_data(
        self,
        raster_stack: Union[RasterStack, List[str]],
        sites: 'geopandas.GeoDataFrame',  # type: ignore
        n_background: int = 1000,
        samples_per_location: int = 20,
        background_exclusion_buffer: Optional[float] = None,
        site_buffer: Optional[float] = None
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
        from ..io.vector import extract_at_points, generate_background_points
        
        if isinstance(raster_stack, list):
            raster_stack = RasterStack.from_files(raster_stack)
        
        # Extract site samples
        site_collections = extract_at_points(
            raster_stack,
            sites,
            buffer_radius=site_buffer,
            n_samples=samples_per_location,
            random_seed=self.seed
        )
        for coll in site_collections:
            coll.label = 1
        
        # Generate exclusion geometries if buffer specified
        exclusion_geoms = None
        if background_exclusion_buffer is not None:
            exclusion_geoms = [
                site.buffer(background_exclusion_buffer) 
                for site in sites.geometry
            ]
        
        # Generate background points
        background_points = generate_background_points(
            raster_stack,
            exclusion_geoms=exclusion_geoms,
            n_points=n_background,
            random_seed=self.seed
        )
        
        # Extract background samples
        background_collections = extract_at_points(
            raster_stack,
            background_points,
            n_samples=samples_per_location,
            random_seed=self.seed
        )
        for coll in background_collections:
            coll.label = 0
        
        return TrainingData(
            collections=site_collections + background_collections,
            feature_names=raster_stack.band_names,
            crs=raster_stack.crs
        )
    
    def fit(self, training_data: TrainingData) -> 'KLRfome':
        """
        Fit the KLRfome model.
        
        Parameters:
            training_data: Prepared TrainingData object
        
        Returns:
            self (for method chaining)
        """
        self._training_data = training_data
        
        # Build similarity matrix
        self._similarity_matrix = self._distribution_kernel.build_similarity_matrix(
            training_data.collections
        )
        
        # Fit KLR
        if self._klr is None:
            self._klr = KernelLogisticRegression(lambda_reg=self.lambda_reg)
        
        self._fit_result = self._klr.fit(
            self._similarity_matrix,
            training_data.labels
        )
        
        if not self._fit_result.converged:
            warnings.warn("KLR fitting did not converge")
        
        return self
    
    def predict(
        self,
        raster_stack: Union[RasterStack, List[str]],
        batch_size: int = 1000,
        show_progress: bool = True
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
        if self._fit_result is None:
            raise RuntimeError("Model must be fit before prediction")
        if self._training_data is None:
            raise RuntimeError("Training data not available")
        if self._distribution_kernel is None:
            raise RuntimeError("Distribution kernel not initialized")
        
        if isinstance(raster_stack, list):
            raster_stack = RasterStack.from_files(raster_stack)
        
        predictor = FocalPredictor(
            distribution_kernel=self._distribution_kernel,
            klr_alpha=self._fit_result.alpha,
            training_data=self._training_data,
            window_size=self.window_size
        )
        
        return predictor.predict_raster(
            raster_stack,
            batch_size=batch_size,
            show_progress=show_progress
        )
    
    def save_prediction(
        self,
        path: str,
        prediction: jnp.ndarray,
        reference_raster: Optional[RasterStack] = None
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
                "reference_raster must be provided. "
                "Use the raster stack used for prediction."
            )
        
        # Create a temporary RasterStack with prediction data
        # We'll use the reference raster's metadata
        temp_stack = RasterStack(
            data=jnp.expand_dims(prediction, axis=0),  # Add band dimension
            transform=reference_raster.transform,
            crs=reference_raster.crs,
            band_names=["prediction"],
            nodata=None
        )
        
        temp_stack.save(path, data=prediction)
    
    def cross_validate(
        self,
        training_data: TrainingData,
        n_folds: int = 5,
        stratified: bool = True
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
        from ..utils.validation import cross_validate
        return cross_validate(self, training_data, n_folds, stratified, self.seed)
    
    def _generate_background_points(self, *args, **kwargs):
        """Generate random background sample locations."""
        # Implementation is now in io.vector.generate_background_points
        from ..io.vector import generate_background_points
        return generate_background_points(*args, **kwargs)

