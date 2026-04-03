# OIRseg — Retinal Image Segmentation

## Environment
- OS: Windows 10; Claude Code uses bash — use Unix shell syntax and forward slashes
- Package manager: **uv** (not pip, not poetry). Run commands with `uv run`
- Config format: PEP 621 in pyproject.toml — never add `[tool.poetry]` or poetry build-system fields
- Python >=3.10 (see `requires-python` in pyproject.toml)
- Linting: ruff (check + format), configured in pyproject.toml `[tool.ruff]`
- Pre-commit: ruff v0.11.4 with `--fix`, plus trailing-whitespace / end-of-file-fixer / check-yaml / check-added-large-files (max 500 KB)
- Tests: `uv run pytest` (configured in pyproject.toml `[tool.pytest.ini_options]`)
- App: `streamlit run app.py`
- Model checkpoint: `Model V4 output/checkpoints/best_model.pth`

## Code Changes
- When asked for UI or display-level changes, only modify display strings and labels — do NOT rename internal variables, config keys, or function names unless explicitly asked
- Respect the scope of what is asked — do not "improve" adjacent code, add docstrings, or refactor unless requested
- Do not add features, error handling, or abstractions beyond what was asked

## Domain Knowledge
- This project segments retinal flatmount images into 3 zones: **NV** (neovascular), **VO** (vaso-obliterated), **retina**
- Mask order in `Config.mask_names`: `("nv", "vo", "retina")`
- **NV% and vessel% denominator = retinal area** (union of all non-background masks), NOT full image pixel count
- Background mask = non-retinal area (outside the retinal tissue boundary)
- This follows the Connor et al. 2009 standard for OIR quantification

## Visualization
- Overlay functions already exist in `src/predict.py`: `save_overlay()` and `save_overlay_large()`
- Always check these before creating new overlay/visualization code
- App overlay colors (app.py): NV=red, VO=white, retina=blue
- Predict overlay colors (src/predict.py): NV=purple, VO=blue, retina=green
- Base image uses grayscale from red channel (fluorescence images)

## Debugging Guidelines
- Read the FULL file/notebook before making any changes — do not patch the first error you see
- Identify ALL errors before applying fixes; check for dependencies between errors
- Before writing any patch, mentally verify it against: device mismatches (CPU/CUDA), missing imports, Windows path issues, worker subprocess isolation, BOM encoding (`utf-8-sig`)
- Prefer minimal targeted patches over monkey-patching or wholesale rewrites

## Project Structure
- `src/config.py` — Config dataclass: model hyperparams, mask_names, thresholds, post-processing
- `src/model.py` — U-Net architecture (segmentation-models-pytorch, EfficientNet-B4 encoder)
- `src/predict.py` — inference pipeline, post-processing, overlay generation
- `config.py` — RAG configuration (API keys, ChromaDB, PubMed settings) — different from src/config.py
- `app.py` — Streamlit web interface
- `rag.py` — PubMed RAG retrieval + Claude Q&A
- `ingest.py` — PubMed abstract ingestion into ChromaDB
- `theme.py` — Streamlit dark gold glassmorphism theme
- `tests/` — pytest tests (test_config, test_ingest, test_predict_utils)
