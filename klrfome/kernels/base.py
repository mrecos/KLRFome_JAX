"""Base kernel protocols and interfaces."""

from typing import Protocol, runtime_checkable
from jaxtyping import Array, Float


@runtime_checkable
class Kernel(Protocol):
    """Protocol for kernel functions."""
    
    def __call__(
        self, 
        X: Float[Array, "n d"], 
        Y: Float[Array, "m d"]
    ) -> Float[Array, "n m"]:
        """
        Compute kernel matrix between X and Y.
        
        Parameters:
            X: First set of points, shape (n, d)
            Y: Second set of points, shape (m, d)
        
        Returns:
            Kernel matrix of shape (n, m)
        """
        ...
    
    @property
    def sigma(self) -> float:
        """Kernel bandwidth parameter."""
        ...


@runtime_checkable
class ApproximateKernel(Protocol):
    """Protocol for kernels with explicit feature maps."""
    
    def feature_map(
        self, 
        X: Float[Array, "n d"]
    ) -> Float[Array, "n D"]:
        """
        Map inputs to approximate feature space.
        
        Parameters:
            X: Input points, shape (n, d)
        
        Returns:
            Feature map of shape (n, D) where D is the feature dimension
        """
        ...

