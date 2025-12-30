"""Core data structures for KLRfome."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np


@dataclass
class SampleCollection:
    """
    Represents samples from a single site or background location.
    
    Attributes:
        samples: Array of shape (n_samples, n_features)
        label: 1 for site, 0 for background
        id: Unique identifier for this location
        metadata: Optional dict for additional info (coordinates, etc.)
    """
    samples: Float[Array, "n_samples n_features"]
    label: Literal[0, 1]
    id: str
    metadata: Optional[Dict] = None
    
    @property
    def n_samples(self) -> int:
        """Number of samples in this collection."""
        return self.samples.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features per sample."""
        return self.samples.shape[1]
    
    def mean_embedding(self) -> Float[Array, "n_features"]:
        """Compute the empirical mean of samples."""
        return jnp.mean(self.samples, axis=0)


@dataclass
class TrainingData:
    """
    Complete training dataset for KLRfome.
    
    Attributes:
        collections: List of SampleCollection objects
        feature_names: Names of the environmental variables
        crs: Coordinate reference system (from rasterio/pyproj)
    """
    collections: List[SampleCollection]
    feature_names: List[str]
    crs: Optional[str] = None
    
    @property
    def n_locations(self) -> int:
        """Total number of locations (sites + background)."""
        return len(self.collections)
    
    @property
    def labels(self) -> Float[Array, "n_locations"]:
        """Array of labels (1 for site, 0 for background)."""
        return jnp.array([c.label for c in self.collections])
    
    @property
    def n_sites(self) -> int:
        """Number of site locations."""
        return sum(1 for c in self.collections if c.label == 1)
    
    @property
    def n_background(self) -> int:
        """Number of background locations."""
        return sum(1 for c in self.collections if c.label == 0)
    
    def train_test_split(
        self, 
        test_fraction: float = 0.2, 
        stratify: bool = True,
        seed: int = 42
    ) -> Tuple['TrainingData', 'TrainingData']:
        """
        Split into training and testing sets.
        
        Parameters:
            test_fraction: Fraction of data to use for testing
            stratify: Whether to stratify by label
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_data, test_data)
        """
        import random
        
        random.seed(seed)
        np.random.seed(seed)
        
        n = len(self.collections)
        
        if stratify:
            # Separate sites and background
            sites = [c for c in self.collections if c.label == 1]
            background = [c for c in self.collections if c.label == 0]
            
            n_sites = len(sites)
            n_background = len(background)
            
            # Shuffle before splitting
            random.shuffle(sites)
            random.shuffle(background)
            
            # Calculate split sizes proportionally
            # Use round to get better balance for small datasets
            # Ensure at least 1 sample per class in test when possible
            if n_sites > 0:
                n_sites_test = round(n_sites * test_fraction)
                # Ensure we leave at least 1 in training if we have more than 1
                if n_sites > 1:
                    n_sites_test = min(max(1, n_sites_test), n_sites - 1)
                else:
                    n_sites_test = 0
            else:
                n_sites_test = 0
            
            if n_background > 0:
                n_background_test = round(n_background * test_fraction)
                # Ensure we leave at least 1 in training if we have more than 1
                if n_background > 1:
                    n_background_test = min(max(1, n_background_test), n_background - 1)
                else:
                    n_background_test = 0
            else:
                n_background_test = 0
            
            # Split each class
            test_sites = sites[:n_sites_test] if n_sites_test > 0 else []
            train_sites = sites[n_sites_test:] if n_sites_test > 0 else sites
            test_background = background[:n_background_test] if n_background_test > 0 else []
            train_background = background[n_background_test:] if n_background_test > 0 else background
            
            train_collections = train_sites + train_background
            test_collections = test_sites + test_background
        else:
            # Simple random split
            n_test = int(n * test_fraction)
            indices = list(range(n))
            random.shuffle(indices)
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
            
            train_collections = [self.collections[i] for i in train_indices]
            test_collections = [self.collections[i] for i in test_indices]
        
        train_data = TrainingData(
            collections=train_collections,
            feature_names=self.feature_names,
            crs=self.crs
        )
        
        test_data = TrainingData(
            collections=test_collections,
            feature_names=self.feature_names,
            crs=self.crs
        )
        
        return train_data, test_data


@dataclass 
class RasterStack:
    """
    Wrapper around a stack of co-registered rasters.
    
    Provides convenient access for focal window extraction.
    
    Attributes:
        data: Array of shape (n_bands, height, width)
        transform: Affine transformation (from rasterio)
        crs: Coordinate reference system
        band_names: Names of the raster bands
        nodata: No-data value
    """
    data: Float[Array, "n_bands height width"]
    transform: 'rasterio.Affine'  # type: ignore
    crs: str
    band_names: List[str]
    nodata: Optional[float] = None
    
    @property
    def n_bands(self) -> int:
        """Number of bands in the raster stack."""
        return self.data.shape[0]
    
    @property
    def height(self) -> int:
        """Height of the raster in pixels."""
        return self.data.shape[1]
    
    @property
    def width(self) -> int:
        """Width of the raster in pixels."""
        return self.data.shape[2]
    
    @classmethod
    def from_files(cls, file_paths: List[str]) -> 'RasterStack':
        """
        Load from multiple single-band rasters.
        
        Parameters:
            file_paths: List of paths to raster files
        
        Returns:
            RasterStack object
        """
        import rasterio
        
        # Read first raster to get dimensions and metadata
        with rasterio.open(file_paths[0]) as src:
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs.to_string() if src.crs else ""
            nodata = src.nodata
        
        # Read all rasters
        bands = []
        band_names = []
        
        for file_path in file_paths:
            with rasterio.open(file_path) as src:
                band = src.read(1)  # Read first band
                bands.append(band)
                band_names.append(src.descriptions[0] if src.descriptions[0] else file_path)
        
        # Stack bands: (n_bands, height, width)
        data = jnp.array(np.stack(bands, axis=0))
        
        return cls(
            data=data,
            transform=transform,
            crs=crs,
            band_names=band_names,
            nodata=nodata
        )
    
    @classmethod
    def from_multiband(cls, file_path: str) -> 'RasterStack':
        """
        Load from a single multi-band raster.
        
        Parameters:
            file_path: Path to multi-band raster file
        
        Returns:
            RasterStack object
        """
        import rasterio
        
        with rasterio.open(file_path) as src:
            data = jnp.array(src.read())  # Shape: (n_bands, height, width)
            transform = src.transform
            crs = src.crs.to_string() if src.crs else ""
            nodata = src.nodata
            band_names = [
                src.descriptions[i] if src.descriptions and src.descriptions[i] 
                else f"band_{i+1}" 
                for i in range(src.count)
            ]
        
        return cls(
            data=data,
            transform=transform,
            crs=crs,
            band_names=band_names,
            nodata=nodata
        )
    
    def extract_window(
        self, 
        row: int, 
        col: int, 
        size: int
    ) -> Float[Array, "size size n_bands"]:
        """
        Extract an NxN window centered at (row, col).
        
        Parameters:
            row: Row index (center of window)
            col: Column index (center of window)
            size: Window size (must be odd)
        
        Returns:
            Window array of shape (size, size, n_bands)
        """
        if size % 2 == 0:
            raise ValueError("Window size must be odd")
        
        pad = size // 2
        padded = jnp.pad(
            self.data,
            ((0, 0), (pad, pad), (pad, pad)),
            mode='reflect'
        )
        
        # Extract window
        window = padded[:, row:row+size, col:col+size]
        
        # Transpose to (size, size, n_bands)
        return jnp.transpose(window, (1, 2, 0))
    
    def extract_at_points(
        self,
        points: 'geopandas.GeoDataFrame',  # type: ignore
        buffer_radius: Optional[float] = None,
        n_samples: int = 10,
        random_seed: Optional[int] = None
    ) -> List[SampleCollection]:
        """
        Extract samples at point locations, optionally with buffer.
        
        Parameters:
            points: GeoDataFrame with point geometries
            buffer_radius: Optional buffer radius around points
            n_samples: Number of samples to extract per location
            random_seed: Random seed for sampling
        
        Returns:
            List of SampleCollection objects
        """
        from klrfome.io.vector import extract_at_points
        return extract_at_points(
            self, points, buffer_radius, n_samples, random_seed
        )
    
    def save(
        self,
        file_path: str,
        data: Optional[Float[Array, "height width"]] = None,
        band: int = 1
    ):
        """
        Save raster data to GeoTIFF.
        
        Parameters:
            file_path: Output file path
            data: Optional 2D array to save (if None, saves first band of self.data)
            band: Band index to save if data is None
        """
        import rasterio
        from rasterio.crs import CRS
        
        if data is None:
            data_to_save = np.array(self.data[band - 1])  # Convert to numpy
        else:
            data_to_save = np.array(data)
        
        # Get CRS
        crs = CRS.from_string(self.crs) if self.crs else None
        
        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=data_to_save.shape[0],
            width=data_to_save.shape[1],
            count=1,
            dtype=data_to_save.dtype,
            crs=crs,
            transform=self.transform,
            nodata=self.nodata
        ) as dst:
            dst.write(data_to_save, 1)

