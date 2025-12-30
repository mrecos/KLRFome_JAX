"""Distribution-level kernels via mean embeddings."""

import jax.numpy as jnp
from typing import Union, List
from jaxtyping import Array, Float

from .base import Kernel, ApproximateKernel
from .rbf import RBFKernel
from .rff import RandomFourierFeatures
from ..data.formats import SampleCollection


class MeanEmbeddingKernel:
    """
    Kernel on distributions via mean embeddings.
    
    Given two sets of samples X = {x_1, ..., x_m} and Y = {y_1, ..., y_n},
    computes similarity as:
    
    K(X, Y) = (1/mn) Σ_i Σ_j k(x_i, y_j)
    
    This is equivalent to the inner product of mean embeddings in RKHS:
    <μ_X, μ_Y>_H where μ_X = (1/m) Σ_i φ(x_i)
    
    Parameters:
        base_kernel: The point-level kernel (e.g., RBFKernel or RandomFourierFeatures)
    """
    
    def __init__(
        self, 
        base_kernel: Union[RBFKernel, RandomFourierFeatures],
    ):
        self.base_kernel = base_kernel
        self._use_rff = isinstance(base_kernel, RandomFourierFeatures)
    
    def __call__(
        self,
        X: Float[Array, "m d"],
        Y: Float[Array, "n d"]
    ) -> float:
        """
        Compute distribution similarity between sample sets X and Y.
        
        Returns a scalar similarity value.
        
        Parameters:
            X: First set of samples, shape (m, d)
            Y: Second set of samples, shape (n, d)
        
        Returns:
            Scalar similarity value
        """
        if self._use_rff:
            # Efficient: compute mean embeddings in feature space
            # Initialize weights if needed
            if isinstance(self.base_kernel, RandomFourierFeatures):
                self.base_kernel._initialize_weights(X.shape[1])
            
            phi_X = self.base_kernel.feature_map(X)  # (m, D)
            phi_Y = self.base_kernel.feature_map(Y)  # (n, D)
            mean_X = jnp.mean(phi_X, axis=0)  # (D,)
            mean_Y = jnp.mean(phi_Y, axis=0)  # (D,)
            return float(jnp.dot(mean_X, mean_Y))
        else:
            # Exact: compute full kernel matrix and average
            K = self.base_kernel(X, Y)  # (m, n)
            return float(jnp.mean(K))
    
    def build_similarity_matrix(
        self,
        collections: List[SampleCollection]
    ) -> Float[Array, "N N"]:
        """
        Build the N×N similarity matrix between all collections.
        
        This is the core computation for KLR fitting.
        
        Parameters:
            collections: List of SampleCollection objects
        
        Returns:
            Similarity matrix of shape (N, N) where N = len(collections)
        """
        n = len(collections)
        
        if n == 0:
            return jnp.array([]).reshape(0, 0)
        
        if self._use_rff:
            # Efficient batch computation with RFF
            # First, compute mean embeddings for all collections
            mean_embeddings = []
            
            # Initialize weights if needed (use first collection to determine dim)
            if isinstance(self.base_kernel, RandomFourierFeatures):
                first_coll = collections[0]
                self.base_kernel._initialize_weights(first_coll.n_features)
            
            for coll in collections:
                phi = self.base_kernel.feature_map(coll.samples)
                mean_embeddings.append(jnp.mean(phi, axis=0))
            
            mean_embeddings = jnp.stack(mean_embeddings)  # (N, D)
            # Similarity matrix is just dot products of mean embeddings
            return jnp.dot(mean_embeddings, mean_embeddings.T)
        
        else:
            # Exact computation (slower)
            K = jnp.zeros((n, n))
            
            for i in range(n):
                for j in range(i, n):
                    k_ij = self(
                        collections[i].samples, 
                        collections[j].samples
                    )
                    K = K.at[i, j].set(k_ij)
                    if i != j:
                        K = K.at[j, i].set(k_ij)  # Symmetry
            
            return K

