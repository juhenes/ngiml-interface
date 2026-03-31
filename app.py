from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import tempfile
from threading import Lock
from time import time
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from PIL import Image

from src import load_model_from_checkpoint
from src.runtime import run_inference_with_model

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

CHECKPOINT_OPTIONS = [
    ("multi-dataset-full", "Multi Dataset Full", "multi-dataset-full.pt"),
    ("casia-full", "CASIA Full", "casia-full.pt"),
    ("casia-effnet-swin", "CASIA Effnet + Swin", "casia-effnet+swin.pt"),
    ("casia-swin", "CASIA Swin", "casia-swin.pt"),
    ("casia-effnet", "CASIA Effnet", "casia-effnet.pt"),
]

PREVIEW_TTL_SECONDS = 300
MODEL_CACHE_LIMIT = 2


@dataclass
class CachedModel:
    model: object
    device: object
    checkpoint_info: dict
    checkpoint_path: Path


@dataclass
class PreviewAsset:
    content: bytes
    expires_at: float


app = FastAPI(title="NGIML Interface", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_model_cache: OrderedDict[str, CachedModel] = OrderedDict()
_model_cache_lock = Lock()
_preview_store: dict[str, PreviewAsset] = {}
_preview_store_lock = Lock()


def list_checkpoints() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for value, label, filename in CHECKPOINT_OPTIONS:
        checkpoint_path = CHECKPOINTS_DIR / filename
        if checkpoint_path.exists():
            items.append(
                {
                    "value": value,
                    "label": label,
                    "filename": filename,
                }
            )
    return items


def resolve_checkpoint(selection: str) -> tuple[Path, str] | None:
    for value, label, filename in CHECKPOINT_OPTIONS:
        if selection != value:
            continue
        checkpoint_path = (CHECKPOINTS_DIR / filename).resolve()
        if checkpoint_path.exists() and checkpoint_path.is_file():
            return checkpoint_path, label
    return None


def _prune_preview_store() -> None:
    now = time()
    expired = [key for key, asset in _preview_store.items() if asset.expires_at <= now]
    for key in expired:
        _preview_store.pop(key, None)


def _png_bytes_from_array(array: np.ndarray) -> bytes:
    image = Image.fromarray(array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _rgb_tensor_bytes(image) -> bytes:
    image_np = (image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return _png_bytes_from_array(image_np)


def _single_channel_tensor_bytes(image) -> bytes:
    image_np = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    return _png_bytes_from_array(image_np)


def _overlay_bytes(image: np.ndarray) -> bytes:
    overlay_np = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _png_bytes_from_array(overlay_np)


def _store_preview_bytes(content: bytes) -> str:
    preview_id = uuid4().hex
    with _preview_store_lock:
        _prune_preview_store()
        _preview_store[preview_id] = PreviewAsset(
            content=content,
            expires_at=time() + PREVIEW_TTL_SECONDS,
        )
    return preview_id


def _get_or_load_model(cache_key: str, checkpoint_path: Path) -> CachedModel:
    with _model_cache_lock:
        cached = _model_cache.get(cache_key)
        if cached is not None:
            _model_cache.move_to_end(cache_key)
            return cached

    model, device, checkpoint_info = load_model_from_checkpoint(checkpoint_path)
    cached_model = CachedModel(
        model=model,
        device=device,
        checkpoint_info=checkpoint_info,
        checkpoint_path=checkpoint_path,
    )

    with _model_cache_lock:
        existing = _model_cache.get(cache_key)
        if existing is not None:
            _model_cache.move_to_end(cache_key)
            return existing
        _model_cache[cache_key] = cached_model
        while len(_model_cache) > MODEL_CACHE_LIMIT:
            _model_cache.popitem(last=False)
    return cached_model


def _run_prediction_from_bytes(selection: str, image_bytes: bytes, suffix: str) -> dict:
    resolved = resolve_checkpoint(selection)
    if resolved is None:
        raise ValueError("Please choose a valid checkpoint.")
    checkpoint_path, checkpoint_label = resolved
    cached = _get_or_load_model(selection, checkpoint_path)

    with tempfile.TemporaryDirectory(prefix="ngiml-web-") as temp_dir:
        image_path = Path(temp_dir) / f"input{suffix}"
        image_path.write_bytes(image_bytes)
        prediction = run_inference_with_model(
            cached.model,
            cached.device,
            cached.checkpoint_info,
            checkpoint_path=cached.checkpoint_path,
            image_path=image_path,
            output_dir=None,
        )

    return {
        "checkpoint_label": checkpoint_label,
        "input_bytes": _rgb_tensor_bytes(prediction["working_image"]),
        "probability_bytes": _single_channel_tensor_bytes(prediction["preview_probability"]),
        "binary_bytes": _single_channel_tensor_bytes(prediction["preview_binary"]),
        "overlay_bytes": _overlay_bytes(prediction["preview_overlay"]),
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    checkpoints = list_checkpoints()
    selected_checkpoint = checkpoints[0]["value"] if checkpoints else None
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "checkpoints": checkpoints,
            "selected_checkpoint": selected_checkpoint,
        },
    )


@app.get("/preview/{preview_id}")
async def preview(preview_id: str) -> Response:
    with _preview_store_lock:
        _prune_preview_store()
        asset = _preview_store.get(preview_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Preview expired.")
    return Response(content=asset.content, media_type="image/png")


@app.post("/predict")
async def predict(
    checkpoint: str = Form(...),
    image: UploadFile = File(...),
) -> JSONResponse:
    if not image.filename:
        return JSONResponse({"ok": False, "error": "Please upload an image."}, status_code=400)

    suffix = Path(image.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        return JSONResponse(
            {"ok": False, "error": "Unsupported image type. Use PNG, JPG, JPEG, BMP, TIF, TIFF, or WEBP."},
            status_code=400,
        )

    try:
        image_bytes = await image.read()
        payload = await run_in_threadpool(_run_prediction_from_bytes, checkpoint, image_bytes, suffix)
        input_id = await run_in_threadpool(_store_preview_bytes, payload["input_bytes"])
        probability_id = await run_in_threadpool(_store_preview_bytes, payload["probability_bytes"])
        binary_id = await run_in_threadpool(_store_preview_bytes, payload["binary_bytes"])
        overlay_id = await run_in_threadpool(_store_preview_bytes, payload["overlay_bytes"])
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Inference failed: {exc}"}, status_code=500)
    finally:
        await image.close()

    return JSONResponse(
        {
            "ok": True,
            "checkpointLabel": payload["checkpoint_label"],
            "previews": {
                "input": f"/preview/{input_id}",
                "prediction": f"/preview/{probability_id}",
                "binary": f"/preview/{binary_id}",
                "overlay": f"/preview/{overlay_id}",
            },
        }
    )
