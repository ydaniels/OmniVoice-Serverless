#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import time
import traceback
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import requests
import runpod
import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice

try:
    import boto3
except ImportError:  # pragma: no cover - installed in Docker image
    boto3 = None


LOGGER = logging.getLogger("omni_serverless")

MODEL_ID = os.getenv("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
MAX_TEXT_LENGTH = int(os.getenv("OMNIVOICE_MAX_TEXT_LENGTH", "1200"))
MAX_REF_AUDIO_MB = int(os.getenv("OMNIVOICE_MAX_REF_AUDIO_MB", "50"))
DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("OMNIVOICE_DOWNLOAD_TIMEOUT_SECONDS", "60"))
LOAD_ASR = (
    os.getenv("OMNIVOICE_LOAD_ASR", "0").strip().lower() in {"1", "true", "yes", "on"}
)
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PREFIX = os.getenv("S3_PREFIX", "omnivoice/outputs").strip().strip("/")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip() or None
S3_REGION = (
    os.getenv("AWS_REGION", "").strip()
    or os.getenv("S3_REGION", "").strip()
    or "us-east-1"
)
S3_CUSTOM_DOMAIN = os.getenv("S3_CUSTOM_DOMAIN", "").strip() or None
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL", "").strip() or None
S3_OBJECT_ACL = os.getenv("S3_OBJECT_ACL", "").strip() or None
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "")).strip()
S3_SECRET_ACCESS_KEY = os.getenv(
    "S3_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "")
).strip()
S3_SESSION_TOKEN = os.getenv("S3_SESSION_TOKEN", os.getenv("AWS_SESSION_TOKEN", "")).strip()

_MODEL: OmniVoice | None = None
_MODEL_LOCK = Lock()


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _get_model() -> OmniVoice:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            device = _detect_device()
            dtype = _dtype_for_device(device)
            LOGGER.info(
                "Loading OmniVoice model=%s device=%s dtype=%s load_asr=%s",
                MODEL_ID,
                device,
                str(dtype),
                LOAD_ASR,
            )
            _MODEL = OmniVoice.from_pretrained(
                MODEL_ID,
                device_map=device,
                dtype=dtype,
                load_asr=LOAD_ASR,
            )
    return _MODEL


def _extract_input(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input", {})
    if not isinstance(payload, dict):
        raise ValueError("event.input must be a JSON object")
    return payload


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def _decode_base64_to_temp_file(encoded_audio: str) -> str:
    data_string = encoded_audio.strip()
    if data_string.startswith("data:") and "," in data_string:
        data_string = data_string.split(",", 1)[1]
    audio_bytes = base64.b64decode(data_string, validate=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with tmp:
        tmp.write(audio_bytes)
    return tmp.name


def _download_to_temp_file(url: str) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix or ".wav"
    max_bytes = MAX_REF_AUDIO_MB * 1024 * 1024
    downloaded = 0
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        with requests.get(url, stream=True, timeout=(10, DOWNLOAD_TIMEOUT_SECONDS)) as response:
            response.raise_for_status()
            with tmp:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(
                            f"Reference audio is too large (> {MAX_REF_AUDIO_MB} MB)."
                        )
                    tmp.write(chunk)
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return tmp.name


def _resolve_ref_audio(payload: dict[str, Any], temp_files: list[str]) -> str | None:
    audio_b64 = payload.get("ref_audio_base64")
    if audio_b64:
        if not isinstance(audio_b64, str):
            raise ValueError("ref_audio_base64 must be a base64 string")
        path = _decode_base64_to_temp_file(audio_b64)
        temp_files.append(path)
        return path

    ref_audio = payload.get("ref_audio")
    if not ref_audio:
        return None
    if not isinstance(ref_audio, str):
        raise ValueError("ref_audio must be a local path or URL string")

    stripped = ref_audio.strip()
    if stripped.startswith("http://") or stripped.startswith("https://"):
        path = _download_to_temp_file(stripped)
        temp_files.append(path)
        return path

    if not Path(stripped).exists():
        raise ValueError(f"ref_audio path does not exist: {stripped}")
    return stripped


def _build_generate_kwargs(payload: dict[str, Any], ref_audio_path: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    if ref_audio_path is not None:
        kwargs["ref_audio"] = ref_audio_path

    for key in ("language", "ref_text", "instruct"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            kwargs[key] = str(value).strip()

    int_fields = ("num_step",)
    for key in int_fields:
        if key in payload and payload[key] is not None:
            kwargs[key] = int(payload[key])

    float_fields = (
        "duration",
        "guidance_scale",
        "speed",
        "t_shift",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
    )
    for key in float_fields:
        if key in payload and payload[key] is not None:
            kwargs[key] = float(payload[key])

    bool_fields: dict[str, bool] = {
        "denoise": True,
        "postprocess_output": True,
    }
    for key, default in bool_fields.items():
        if key in payload and payload[key] is not None:
            kwargs[key] = _as_bool(payload[key], default=default)

    return kwargs


def _audio_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> tuple[bytes, float]:
    mono = audio_tensor.detach().cpu().float()
    if mono.ndim == 2 and mono.shape[0] == 1:
        mono = mono.squeeze(0)
    if mono.ndim != 1:
        raise ValueError(f"Unexpected audio shape: {tuple(mono.shape)}")

    samples = mono.numpy()
    duration_seconds = float(samples.shape[0]) / float(sample_rate)

    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue(), duration_seconds


def _is_s3_enabled() -> bool:
    if not S3_BUCKET:
        return False
    if boto3 is None:
        return False
    return bool(S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY)


def _normalize_public_base_url(value: str) -> str:
    if not value:
        return value
    if "://" not in value:
        return "https://" + value.lstrip("/")
    return value


def _s3_public_url(bucket: str, key: str) -> str:
    if S3_CUSTOM_DOMAIN:
        return _normalize_public_base_url(S3_CUSTOM_DOMAIN).rstrip("/") + "/" + key
    if S3_PUBLIC_BASE_URL:
        return _normalize_public_base_url(S3_PUBLIC_BASE_URL).rstrip("/") + "/" + key
    if S3_ENDPOINT_URL:
        return f"{S3_ENDPOINT_URL.rstrip('/')}/{bucket}/{key}"
    return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"


def _upload_wav_to_s3(wav_bytes: bytes) -> dict[str, str]:
    if boto3 is None:
        raise RuntimeError("boto3 is not installed")

    key_prefix = S3_PREFIX or "omnivoice/outputs"
    date_part = time.strftime("%Y/%m/%d")
    key = f"{key_prefix}/{date_part}/{uuid4().hex}.wav"

    client = boto3.client(
        "s3",
        aws_access_key_id=S3_ACCESS_KEY_ID or None,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY or None,
        aws_session_token=S3_SESSION_TOKEN or None,
        region_name=S3_REGION,
        endpoint_url=S3_ENDPOINT_URL,
    )

    put_kwargs: dict[str, Any] = {
        "Bucket": S3_BUCKET,
        "Key": key,
        "Body": wav_bytes,
        "ContentType": "audio/wav",
    }
    if S3_OBJECT_ACL:
        put_kwargs["ACL"] = S3_OBJECT_ACL
    client.put_object(**put_kwargs)

    public_url = _s3_public_url(S3_BUCKET, key)
    return {
        "s3_bucket": S3_BUCKET,
        "s3_key": key,
        "s3_url": public_url,
        "public_url": public_url,
    }


def handler(event: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    temp_files: list[str] = []

    try:
        payload = _extract_input(event)
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("text is required")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"text length {len(text)} exceeds OMNIVOICE_MAX_TEXT_LENGTH={MAX_TEXT_LENGTH}"
            )
        return_base64 = _as_bool(payload.get("return_base64"), default=True)
        upload_to_s3 = _as_bool(payload.get("upload_to_s3"), default=True)

        model = _get_model()
        ref_audio_path = _resolve_ref_audio(payload, temp_files)
        generate_kwargs = _build_generate_kwargs(payload, ref_audio_path=ref_audio_path)

        with torch.inference_mode():
            audios = model.generate(text=text, **generate_kwargs)

        if not audios:
            raise RuntimeError("No audio returned by model.generate")

        sample_rate = int(model.sampling_rate or 24000)
        wav_bytes, duration_seconds = _audio_to_wav_bytes(audios[0], sample_rate)

        result: dict[str, Any] = {
            "format": "wav",
            "sample_rate": sample_rate,
            "duration_seconds": round(duration_seconds, 3),
            "elapsed_seconds": round(time.time() - started, 3),
            "model_id": MODEL_ID,
            "device": str(model.device),
        }
        if upload_to_s3 and _is_s3_enabled():
            # When S3 upload is available, return URL fields and skip base64 payload.
            result.update(_upload_wav_to_s3(wav_bytes))
        elif return_base64:
            result["audio_base64"] = base64.b64encode(wav_bytes).decode("utf-8")

        return result
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "error": "CUDA out of memory while running OmniVoice inference.",
            "refresh_worker": True,
        }
    except Exception as exc:
        LOGGER.exception("OmniVoice serverless request failed")
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "refresh_worker": False,
        }
    finally:
        for path in temp_files:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    runpod.serverless.start({"handler": handler})
