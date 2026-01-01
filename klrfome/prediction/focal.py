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
            if hasattr(base_kernel, '_W') and base_kernel._W is not None:
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
    
    def _compute_training_embeddings(self) -> Float[Array, "n_train D"]:
        """Precompute mean embeddings for all training collections."""
        embeddings = []
        base_kernel = self.distribution_kernel.base_kernel
        
        # Initialize weights if needed (use first collection)
        if hasattr(base_kernel, '_initialize_weights'):
            first_coll = self.training_data.collections[0]
            base_kernel._initialize_weights(first_coll.n_features)
        
        for coll in self.training_data.collections:
            phi = base_kernel.feature_map(coll.samples)
            embeddings.append(jnp.mean(phi, axis=0))
        
        return jnp.stack(embeddings)
    
    def predict_window(
        self,
        window_samples: Float[Array, "m d"]
    ) -> float:
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
            K_new = jnp.array([
                round(float(self.distribution_kernel(window_samples, coll.samples)), 3)
                for coll in self.training_data.collections
            ])
        
        # Apply KLR prediction: η = K_new @ alpha, then sigmoid
        eta = jnp.dot(K_new, self.klr_alpha)
        return float(1 / (1 + jnp.exp(-eta)))
    
    def predict_raster(
        self,
        raster_stack: RasterStack,
        batch_size: int = 1000,
        show_progress: bool = True
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
        padded_data = jnp.pad(
            raster_stack.data,
            ((0, 0), (pad, pad), (pad, pad)),
            mode='reflect'
        )
        
        # Generate all window center coordinates
        rows, cols = jnp.meshgrid(
            jnp.arange(height),
            jnp.arange(width),
            indexing='ij'
        )
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
                # Exact kernel: use non-JIT path since we need Python objects
                batch_preds = self._predict_batch_exact(
                    padded_data,
                    batch_coords,
                    pad,
                    self.window_size,
                    self.distribution_kernel,
                    self.training_data,
                    self.klr_alpha
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
                    self._rff_n_features
                )
            predictions.append(batch_preds)
        
        predictions = jnp.concatenate(predictions)
        return predictions.reshape(height, width)
    
    @staticmethod
    @partial(jit, static_argnames=['window_size', 'use_rff', 'rff_n_features'])
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
        rff_n_features: Optional[int]
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
            # Extract window (accounting for padding offset)
            window = lax.dynamic_slice(
                padded_data,
                (0, r + pad, c + pad),
                (padded_data.shape[0], window_size, window_size)
            )
            # Reshape to (n_samples, n_features)
            # window is (bands, window_size, window_size)
            window_samples = window.reshape(window.shape[0], -1).T
            
            # Compute prediction
            if use_rff and training_embeddings is not None and rff_W is not None and rff_b is not None and rff_n_features is not None:
                # RFF path: compute mean embedding and dot with training embeddings
                # Project: Wx + b
                projection = jnp.dot(window_samples, rff_W) + rff_b
                phi = jnp.sqrt(2.0 / rff_n_features) * jnp.cos(projection)
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
        klr_alpha: Float[Array, "n_train"]
    ) -> Float[Array, "batch"]:
        """
        Non-JIT batch prediction for exact kernel path.
        
        This is slower than RFF but provides exact kernel computation.
        """
        predictions = []
        
        for coord in coords:
            r, c = int(coord[0]), int(coord[1])
            # Extract window (accounting for padding offset)
            window = padded_data[:, r:r + window_size, c:c + window_size]
            # Reshape to (n_samples, n_features)
            window_samples = window.reshape(window.shape[0], -1).T
            
            # Compute kernel with each training collection
            K_new = jnp.array([
                jnp.round(distribution_kernel(window_samples, coll.samples), 3)
                for coll in training_data.collections
            ])
            
            # Apply KLR
            eta = jnp.dot(K_new, klr_alpha)
            pred = 1 / (1 + jnp.exp(-eta))
            predictions.append(pred)
        
        return jnp.array(predictions)

