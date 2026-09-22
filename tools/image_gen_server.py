# /// script
# requires-python = ">=3.11"
# ///
"""Minimal FastAPI wrapper around the Qwen-Image-2.1 diffusers pipeline.

This is not an LLM server: Qwen/Qwen-Image-2.1 is a text-to-image /
image-editing diffusion model (a Qwen3-VL-8B text encoder + a 7B single-stream
DiT + a VAE, ~32 GB BF16 across the three submodules), served through
diffusers' `QwenImage21Pipeline`, not vLLM. There is no KV cache and no
continuous batching -- one request at a time, each a fixed number of denoise
steps. The whole point of this service is to keep that pipeline behind a
health-checked HTTP endpoint that fits the repo's one-service-at-a-time
pattern (it wants all four R9700s, so it never runs beside the LLM services).

Config comes entirely from the environment (set in compose.yaml):

  IMAGEGEN_MODEL_DIR   local checkpoint dir (default /srv/llm/Qwen/Qwen-Image-2.1)
  IMAGEGEN_OFFLOAD     placement strategy: device_map | cpu_offload | sequential
  IMAGEGEN_DTYPE       bfloat16 (default) | float32  -- see the gfx1201 note below
  IMAGEGEN_PORT        handled by uvicorn/`command:`, not read here

gfx1201 (RDNA4) caveat: this pipeline has no verified ROCm/RDNA4 report. It is
built against the same AMD ROCm torch wheel the rest of the stack uses and
relies on torch SDPA (no FlashAttention). If the first boot faults or produces
NaN/garbage, try IMAGEGEN_DTYPE=float32 and/or IMAGEGEN_OFFLOAD=sequential --
both are documented fallbacks for Qwen-Image on non-CUDA devices.
"""

from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any

import torch
from diffusers import QwenImage21Pipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

MODEL_DIR = os.environ.get("IMAGEGEN_MODEL_DIR", "/srv/llm/Qwen/Qwen-Image-2.1")
OFFLOAD = os.environ.get("IMAGEGEN_OFFLOAD", "device_map").strip().lower()
DTYPE = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(
    os.environ.get("IMAGEGEN_DTYPE", "bfloat16").strip().lower(), torch.bfloat16
)

app = FastAPI(title="qwen-image-2.1")

# The pipeline is loaded once at startup and mutated (moved across devices) by
# the offload setup, so generation is serialized behind a lock -- diffusers
# pipelines are not safe to run concurrently on the same instance.
_pipe: QwenImage21Pipeline | None = None
_lock = threading.Lock()
_ready = threading.Event()


def _load() -> QwenImage21Pipeline:
    kwargs: dict[str, Any] = {"torch_dtype": DTYPE}
    if OFFLOAD == "device_map":
        # Shard the encoder / DiT / VAE across every visible CUDA/HIP device.
        # This is the fast path (no host<->device ping-pong) and matches how
        # the LLM services spread TP=4 over all four cards.
        kwargs["device_map"] = "balanced"
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_DIR, **kwargs)
    if OFFLOAD == "cpu_offload":
        pipe.enable_model_cpu_offload()
    elif OFFLOAD == "sequential":
        pipe.enable_sequential_cpu_offload()
    return pipe


@app.on_event("startup")
def _startup() -> None:
    global _pipe

    def _go() -> None:
        global _pipe
        _pipe = _load()
        _ready.set()

    threading.Thread(target=_go, daemon=True).start()


@app.get("/health")
def health() -> dict[str, Any]:
    if not _ready.is_set():
        # 503 until the ~32 GB of weights are resident; the compose healthcheck
        # retries this and start_period absorbs the load time.
        raise HTTPException(status_code=503, detail="loading")
    return {"status": "ok", "model": MODEL_DIR, "offload": OFFLOAD, "dtype": str(DTYPE)}


class GenRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1328
    height: int = 1328
    steps: int = 40
    guidance_scale: float = 4.0
    seed: int | None = None
    response_format: str = "url"  # "url" -> image/png bytes; "b64_json" -> OpenAI-style


def _render(req: GenRequest) -> bytes:
    gen = None
    if req.seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(req.seed))
    out = _pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or None,
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
        true_cfg_scale=req.guidance_scale,
        generator=gen,
    )
    buf = io.BytesIO()
    out.images[0].save(buf, format="PNG")
    return buf.getvalue()


@app.post("/generate")
def generate(req: GenRequest) -> Response:
    if _pipe is None:
        raise HTTPException(status_code=503, detail="loading")
    with _lock:  # one diffusion at a time
        try:
            png = _render(req)
        except Exception as exc:  # surface the real device fault to the client
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return Response(content=png, media_type="image/png")


@app.post("/v1/images/generations")
def openai_images(req: GenRequest) -> JSONResponse:
    """OpenAI-compatible images endpoint so litellm can route to it. Returns
    base64 PNG (diffusers has no hosted-image URL to return)."""
    if _pipe is None:
        raise HTTPException(status_code=503, detail="loading")
    with _lock:
        try:
            png = _render(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return JSONResponse(
        {"created": 0, "data": [{"b64_json": base64.b64encode(png).decode()}]}
    )
