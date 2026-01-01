# Example Data

This folder contains synthetic example data for testing KLRfome.

## Files

- `var1.tif`, `var2.tif`, `var3.tif` - Environmental variable rasters (200x200 pixels)
- `sites.geojson` - 25 simulated site point locations
- `metadata.json` - Dataset metadata

## Data Description

The rasters represent synthetic environmental variables with spatial autocorrelation.
Site locations were placed preferentially in areas with specific environmental characteristics,
creating a learnable signal for the model.

## Usage

See the Quick Start section in the main README.md or run `notebooks/01_quickstart.ipynb`.

```python
from klrfome import KLRfome, RasterStack
import geopandas as gpd

# Load example data
raster_stack = RasterStack.from_files([
    'example_data/var1.tif',
    'example_data/var2.tif', 
    'example_data/var3.tif'
])
sites = gpd.read_file('example_data/sites.geojson')

# Fit model
model = KLRfome(sigma=0.5, lambda_reg=0.1, window_size=5)
training_data = model.prepare_data(raster_stack, sites, n_background=50)
model.fit(training_data)

# Predict
predictions = model.predict(raster_stack)
```

