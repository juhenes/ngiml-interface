# NGIML - Interface

FastAPI web interface for running NGIML inference from local checkpoints.

## Quick start

Recommended on Windows: use the included setup script. It creates a clean Python 3.13 virtual environment, installs the web dependencies, and installs CPU PyTorch from the official wheel index.

```bash
.\setup_windows.ps1
```

Then run the app:

```bash
.\run_web.ps1
```

Open `http://127.0.0.1:8000`

## Model setup

Put one or more `.pt` checkpoints inside `checkpoints/`.

## Manual setup

If you prefer manual commands, use plain CPython on Windows and avoid mixing Conda Python with the project virtual environment.

```bash
py -3.13 -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install --upgrade pip
.\.venv313\Scripts\python.exe -m pip install fastapi uvicorn python-multipart jinja2 timm numpy pillow matplotlib
.\.venv313\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv313\Scripts\python.exe -m uvicorn app:app --reload
```

## What the web UI does

- Auto-discovers every `.pt` file inside `checkpoints/`
- Lets the user choose a checkpoint from a dropdown
- Lets the user upload an image
- Runs inference using the existing runtime in `src/`
- Uses `resize_keep_aspect_center_crop` preprocessing by default before inference
- Shows the cropped inference preview for the input image, probability map, binary mask, and overlay
- Does not persist uploads or generated results after processing finishes

## Developer notes

- Main FastAPI entrypoint: `app.py`
- HTML template: `templates/index.html`
- Styles: `static/styles.css`
- Existing CLI inference via `predict.py` still works
- `setup_windows.ps1` and `run_web.ps1` are the recommended workflow on Windows
- The default crop size comes from checkpoint metadata `input_size` when available, otherwise `448`
- The preprocessing flow resizes while keeping aspect ratio, then center-crops to the target size
- The web page includes a loading overlay during inference so long-running requests feel responsive
