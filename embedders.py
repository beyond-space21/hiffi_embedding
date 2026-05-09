import json
import io
import os
import numpy as np
import re
import subprocess
import time
import uuid
from contextlib import nullcontext
from typing import Any
from typing import Dict
from typing import Tuple

import librosa
import torch
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams
from transformers import AutoModel
from transformers import Wav2Vec2FeatureExtractor

from config import settings

os.environ["TRANSFORMERS_CACHE"] = settings.CACHE_DIR
os.environ["HF_HOME"] = settings.CACHE_DIR
# Suppress non-critical transformers advisory warnings in worker logs.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MERT_MODEL_NAME = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE = 24000

mert_model = AutoModel.from_pretrained(
    MERT_MODEL_NAME,
    cache_dir=settings.CACHE_DIR,
    trust_remote_code=True,
).to(DEVICE)
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
    MERT_MODEL_NAME,
    cache_dir=settings.CACHE_DIR,
    trust_remote_code=True,
)
MERT_DIM = int(mert_model.config.hidden_size)

qdrant = QdrantClient(settings.QDRANT_URL)
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
mert_model.eval()

_POINT_ID_NS = uuid.UUID("a3f8c2e1-6b4d-5e9f-8c0d-1a2b3c4d5e6f")
_SAMPLE_RATE = 48000


def ensure_collection() -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if settings.COLLECTION_NAME in existing:
        return

    qdrant.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config={
            "mert": VectorParams(size=MERT_DIM, distance=Distance.COSINE),
        },
    )


def runtime_device_label() -> str:
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        return f"GPU (cuda:0, {gpu_name})"
    return "CPU"


def _as_feature_tensor(emb: Any) -> torch.Tensor:
    if isinstance(emb, torch.Tensor):
        return emb
    if getattr(emb, "pooler_output", None) is not None:
        return emb.pooler_output
    return emb.last_hidden_state[:, 0]


def normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True)


def _autocast_context():
    if DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _chunks(seq: list, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def _upsert_points_batched(points: list[dict]) -> None:
    batch_size = max(1, settings.QDRANT_UPSERT_BATCH_SIZE)
    for batch in _chunks(points, batch_size):
        _upsert_with_fallback(batch)


def _upsert_with_fallback(batch: list[dict]) -> None:
    if not batch:
        return
    try:
        qdrant.upsert(collection_name=settings.COLLECTION_NAME, points=batch)
    except Exception as exc:
        msg = str(exc).lower()
        too_large = "payload error" in msg or "larger than allowed" in msg
        if too_large and len(batch) > 1:
            mid = len(batch) // 2
            _upsert_with_fallback(batch[:mid])
            _upsert_with_fallback(batch[mid:])
            return
        raise


def embed_audio(audio_chunk: Any) -> list[float]:
    inputs = mert_processor(
        [audio_chunk],
        sampling_rate=MERT_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    with torch.inference_mode(), _autocast_context():
        outputs = mert_model(**inputs)
        hidden = outputs.last_hidden_state
        attn = inputs.get("attention_mask")
        if attn is not None and attn.dim() == 2 and attn.shape[1] == hidden.shape[1]:
            mask = attn.unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            # MERT downsamples in time; mask may stay at raw sample length.
            emb = hidden.mean(dim=1)
    return normalize(emb)[0].cpu().tolist()


def embed_audio_chunks(audio_chunks: list[Any]) -> list[list[float]]:
    if not audio_chunks:
        return []
    inputs = mert_processor(
        audio_chunks,
        sampling_rate=MERT_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    with torch.inference_mode(), _autocast_context():
        outputs = mert_model(**inputs)
        hidden = outputs.last_hidden_state
        attn = inputs.get("attention_mask")
        if attn is not None and attn.dim() == 2 and attn.shape[1] == hidden.shape[1]:
            mask = attn.unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            # MERT downsamples in time; mask may stay at raw sample length.
            emb = hidden.mean(dim=1)
    return normalize(emb).cpu().tolist()


def _synthesize_query_audio(query_text: str) -> np.ndarray | None:
    if not settings.OPENAI_AUDIO_BRIDGE_ENABLED:
        return None
    text = query_text.strip()
    if not text:
        return None
    try:
        try:
            response = openai_client.audio.speech.create(
                model=settings.OPENAI_AUDIO_BRIDGE_MODEL,
                voice=settings.OPENAI_AUDIO_BRIDGE_VOICE,
                input=text,
                response_format="wav",
            )
        except TypeError:
            response = openai_client.audio.speech.create(
                model=settings.OPENAI_AUDIO_BRIDGE_MODEL,
                voice=settings.OPENAI_AUDIO_BRIDGE_VOICE,
                input=text,
            )
        if not response.content:
            return None
        audio, _ = librosa.load(io.BytesIO(response.content), sr=MERT_SAMPLE_RATE, mono=True)
        return audio
    except Exception as exc:
        print(f"OpenAI audio bridge failed, skipping MERT query vector: {exc}")
        return None


def parse_query(query: str) -> Dict[str, str]:
    tokens = re.findall(r"\[(visual|audio)\](.*?)(?=\[(?:visual|audio)\]|$)", query, flags=re.I | re.S)
    parsed: Dict[str, str] = {"visual": "", "audio": "", "raw": query.strip()}
    for kind, value in tokens:
        parsed[kind.lower()] = value.strip()
    return parsed


def optimize_query_parts(query: str) -> Dict[str, str]:
    parsed = parse_query(query)
    if not settings.OPENAI_QUERY_OPTIMIZATION:
        return parsed

    system_prompt = (
        "You optimize user video-search queries for audio retrieval.\n"
        "Return JSON only with key: audio.\n"
        "audio: concise sound/music/voice description for audio retrieval.\n"
        "If user intent lacks audio details, infer realistic generic hints.\n"
        "Do not include markdown."
    )

    try:
        response = openai_client.chat.completions.create(
            model=settings.OPENAI_QUERY_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User query: {query}\n"
                        f"Existing parsed hints (may be empty): {json.dumps(parsed)}"
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content or "{}"
        optimized = json.loads(raw)
        audio = str(optimized.get("audio", "")).strip()
        return {"visual": "", "audio": audio, "raw": query.strip()}
    except Exception as exc:
        print(f"OpenAI query optimization failed, using raw query: {exc}")
        return parsed


def embed_query(parts: Dict[str, str]) -> Tuple[list[float], list[float] | None]:
    audio_text = parts.get("audio") or parts.get("raw", "")
    bridge_audio = _synthesize_query_audio(audio_text)
    if bridge_audio is None or len(bridge_audio) == 0:
        return [], None
    return [], embed_audio(bridge_audio)


def _point_id(video_id: str, kind: str, key: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NS, f"{video_id}:{kind}:{key}"))


def _load_audio_ffmpeg(path: str, sample_rate: int = _SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not available in PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed to decode audio '{path}': {stderr}") from exc

    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg decoded zero samples for '{path}'")
    return audio


def load_audio(audio_path: str, sample_rate: int = _SAMPLE_RATE):
    if audio_path.lower().endswith(".aac"):
        return _load_audio_ffmpeg(audio_path, sample_rate=MERT_SAMPLE_RATE)
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    return audio


def index_video(audio_path: str, metadata: Dict, video_id: str) -> int:
    t0 = time.perf_counter()
    audio_points = []

    # Audio (CLAP) - 3-second chunks, batched
    audio_load_start = time.perf_counter()
    audio = load_audio(audio_path, sample_rate=MERT_SAMPLE_RATE)
    audio_load_time = time.perf_counter() - audio_load_start

    chunk_size = MERT_SAMPLE_RATE * 3
    audio_chunks = []
    for offset in range(0, len(audio), chunk_size):
        audio_chunks.append((offset, audio[offset : offset + chunk_size]))

    audio_embed_start = time.perf_counter()
    for batch in _chunks(audio_chunks, max(1, settings.AUDIO_BATCH_SIZE)):
        batch_offsets = [item[0] for item in batch]
        batch_audio = [item[1] for item in batch]
        batch_embeddings = embed_audio_chunks(batch_audio)
        for offset, emb in zip(batch_offsets, batch_embeddings):
            audio_points.append(
                {
                    "id": _point_id(video_id, "audio", str(offset)),
                    "vector": {"mert": emb},
                    "payload": {
                        "type": "audio",
                        "video_id": video_id,
                        "timestamp": offset / MERT_SAMPLE_RATE,
                        "group_by": "video_id",
                        **metadata,
                    },
                }
            )
    if settings.AUDIO_SUMMARY_VECTOR and audio_points:
        audio_matrix = np.asarray([p["vector"]["mert"] for p in audio_points], dtype=np.float32)
        audio_mean = audio_matrix.mean(axis=0)
        audio_norm = np.linalg.norm(audio_mean)
        if audio_norm > 0:
            audio_mean = audio_mean / audio_norm
        audio_points.append(
            {
                "id": _point_id(video_id, "audio", "summary"),
                "vector": {"mert": audio_mean.tolist()},
                "payload": {
                    "type": "audio_summary",
                    "video_id": video_id,
                    "chunk_count": len(audio_chunks),
                    "group_by": "video_id",
                    **metadata,
                },
            }
        )
    audio_embed_time = time.perf_counter() - audio_embed_start

    points = audio_points
    upsert_start = time.perf_counter()
    if points:
        _upsert_points_batched(points)
    upsert_time = time.perf_counter() - upsert_start

    total_time = time.perf_counter() - t0
    print(
        "Indexed video "
        f"{video_id} ({len(points)} points) | "
        f"audio_load={audio_load_time:.2f}s | "
        f"audio_emb={audio_embed_time:.2f}s | "
        f"upsert={upsert_time:.2f}s | total={total_time:.2f}s"
    )
    return len(points)
