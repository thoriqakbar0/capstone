# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: Marimo notebook that hosts the property-valuation UI and pipeline orchestration. Keep new logic in dedicated helper functions within this file or import from future modules under `src/`.
- `property_dataset.csv`: Source data for LightGBM training; treat as read-only and document any schema edits.
- `pyproject.toml` and `uv.lock`: Dependency and interpreter pinning (Python 3.12, `marimo[recommended]`). Update via `uv add` or `uv remove` to keep the lockfile in sync.
- `README.md`: High-level overview—extend with user-facing instructions when features change.

## Build, Test, and Development Commands
- `uv run marimo edit main.py`: Opens the notebook editor with live reactivity; use during UI development.
- `uv run marimo run main.py`: Launches the compiled app for manual verification.
- `uv run python scripts/train.py`: Recommended pattern for future automation; add training scripts under `scripts/` and keep them CLI-friendly.

## Coding Style & Naming Conventions
- Python modules use 4-space indentation, snake_case for variables/functions, PascalCase for classes.
- Prefer pure helper functions placed above the Marimo cells; keep notebook cells concise and side-effect free.
- Document new configuration values or environment variables in `README.md` and inline comments sparingly for non-obvious logic.

## Testing Guidelines
- Adopt `pytest` under a `tests/` directory; mirror notebook behaviors with unit tests around pipeline helpers.
- Name test files `test_<feature>.py` and ensure fixtures cover typical and edge property records.
- Target fast, deterministic tests by stubbing I/O and using small CSV samples; reserve full-data runs for manual checks.

## Commit & Pull Request Guidelines
- Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`) to communicate intent clearly.
- Reference related tickets in the commit body; include brief context on dataset versions or model tweaks.
- Pull requests should summarize changes, list testing steps (`marimo run`, `pytest`), and attach screenshots or GIFs of the UI when modified.
