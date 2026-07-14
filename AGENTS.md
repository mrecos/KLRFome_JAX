# Repository Guidelines

## Project Structure & Module Organization
- `klrfome/` contains the Python/JAX implementation, organized into `data/`, `io/`, `kernels/`, `models/`, `prediction/`, `utils/`, and `visualization/`.
- `tests/` holds pytest suites (unit + integration), with shared fixtures in `tests/conftest.py`.
- `example_data/` and `site_data/` provide sample rasters and vector data used in docs and notebooks.
- `benchmarks/` and `notebooks/` contain performance experiments and exploratory analysis.
- Project metadata and tool config live in `pyproject.toml`.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode.
- `pip install -e ".[dev]"` installs development and test tooling.
- `./run_tests.sh` runs tests with coverage, creating an `htmlcov/` report.
- `./run_tests_simple.sh` runs tests using the venv Python directly if the main script fails.
- `python -m pytest tests/ -v` runs the full test suite manually.

## Coding Style & Naming Conventions
- Python 3.9+ codebase; prefer type hints where it improves clarity.
- Formatting: Black with line length 100 (`python -m black klrfome tests`).
- Linting: Ruff with line length 100 (`python -m ruff check klrfome tests`).
- Type checks: Mypy (`python -m mypy klrfome`).
- Use snake_case for functions/variables, CapWords for classes, and `test_*.py` for test files.

## Testing Guidelines
- Framework: pytest with optional coverage via pytest-cov.
- Naming: tests live in `tests/` and follow `test_*.py` with `test_*` functions.
- Coverage: run `./run_tests.sh` or `pytest tests/ -v --cov=klrfome --cov-report=html`.

## Commit & Pull Request Guidelines
- Commits in this repo use short, imperative summaries (e.g., "Add X", "Fix Y").
- Keep commits focused on a single logical change.
- PRs should include a clear description, linked issue if applicable, and test results.
- If you add features or fix bugs, add or update tests in `tests/`.

## Security & Configuration Tips
- JAX GPU support requires a CUDA-enabled build; see `README.md` for install notes.
- Use a local virtual environment in `venv/` and keep it out of version control.
