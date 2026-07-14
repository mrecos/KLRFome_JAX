# Synthetic example data

This directory contains a deterministic, self-contained dataset for the KLRfome README workflow
and automated smoke tests.

## Files

- `var1.tif`, `var2.tif`, and `var3.tif`: three aligned 200 × 200 environmental rasters;
- `sites.geojson`: 25 simulated site points; and
- `metadata.json`: generation settings, extent, CRS, and random seed.

The rasters contain spatially autocorrelated synthetic variables. Site locations were generated
with a learnable environmental association. The data demonstrate software behavior only and are
not evidence for selecting among M0–M3.

Run the CI-sized end-to-end workflow from the repository root:

```bash
python examples/readme_quickstart.py
```

Run focal prediction over the complete example extent with:

```bash
python examples/readme_quickstart.py --full-surface
```

The fitted example is a presence-background design, so its output is a relative-suitability
surface rather than an occurrence-probability surface. See the main [README](../README.md) and
[model/data foundation](../MODEL_DATA_FOUNDATION.md) for the current API and interpretation.
