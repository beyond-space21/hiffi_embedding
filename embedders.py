import json
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
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams
from transformers import ClapModel
from transformers import ClapProcessor
from transformers import CLIPModel
from transformers import CLIPProcessor

from config import settings

os.environ["TRANSFORMERS_CACHE"] = settings.CACHE_DIR
os.environ["HF_HOME"] = settings.CACHE_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== LOAD MODELS ======================
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir=settings.CACHE_DIR,
).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir=settings.CACHE_DIR,
)

clap_model = ClapModel.from_pretrained(
    "laion/clap-htsat-unfused",
    cache_dir=settings.CACHE_DIR,
).to(DEVICE)
clap_processor = ClapProcessor.from_pretrained(
    "laion/clap-htsat-unfused",
    cache_dir=settings.CACHE_DIR,
)

qdrant = QdrantClient(settings.QDRANT_URL)
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
clip_model.eval()
clap_model.eval()

_POINT_ID_NS = uuid.UUID("a3f8c2e1-6b4d-5e9f-8c0d-1a2b3c4d5e6f")
_SAMPLE_RATE = 48000


def ensure_collection() -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if settings.COLLECTION_NAME in existing:
        return

    qdrant.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config={
            "clip": VectorParams(size=512, distance=Distance.COSINE),
            "clap": VectorParams(size=512, distance=Distance.COSINE),
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


def embed_image(image: Image.Image) -> list[float]:
    inputs = clip_processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode(), _autocast_context():
        emb = _as_feature_tensor(clip_model.get_image_features(**inputs))
    return normalize(emb)[0].cpu().tolist()


def embed_images(images: list[Image.Image]) -> list[list[float]]:
    if not images:
        return []
    inputs = clip_processor(images=images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode(), _autocast_context():
        emb = _as_feature_tensor(clip_model.get_image_features(**inputs))
    return normalize(emb).cpu().tolist()


def embed_audio(audio_chunk: Any) -> list[float]:
    inputs = clap_processor(audio=[audio_chunk], sampling_rate=48000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode(), _autocast_context():
        emb = _as_feature_tensor(clap_model.get_audio_features(**inputs))
    return normalize(emb)[0].cpu().tolist()


def embed_audio_chunks(audio_chunks: list[Any]) -> list[list[float]]:
    if not audio_chunks:
        return []
    inputs = clap_processor(audio=audio_chunks, sampling_rate=48000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode(), _autocast_context():
        emb = _as_feature_tensor(clap_model.get_audio_features(**inputs))
    return normalize(emb).cpu().tolist()


def embed_text_clip(text: str) -> list[float]:
    inputs = clip_processor(text=[text], return_tensors="pt", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode(), _autocast_context():
        emb = _as_feature_tensor(clip_model.get_text_features(**inputs))
    return normalize(emb)[0].cpu().tolist()


def embed_text_clap(text: str) -> list[float]:
    inputs = clap_processor(text=[text], return_tensors="pt", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode(), _autocast_context():
        emb = _as_feature_tensor(clap_model.get_text_features(**inputs))
    return normalize(emb)[0].cpu().tolist()


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
        "You optimize user video-search queries for dual encoders.\n"
        "Return JSON only with keys: visual, audio.\n"
        "visual: concise scene/object/action description for image retrieval.\n"
        "audio: concise sound/music/voice description for audio retrieval.\n"
        "If user intent lacks modality details, infer realistic generic hints.\n"
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
        visual = str(optimized.get("visual", "")).strip()
        audio = str(optimized.get("audio", "")).strip()
        return {"visual": visual, "audio": audio, "raw": query.strip()}
    except Exception as exc:
        print(f"OpenAI query optimization failed, using raw query: {exc}")
        return parsed


def embed_query(parts: Dict[str, str]) -> Tuple[list[float], list[float]]:
    visual_text = parts.get("visual") or parts.get("raw", "")
    audio_text = parts.get("audio") or parts.get("raw", "")
    clip_vec = embed_text_clip(visual_text)
    clap_vec = embed_text_clap(audio_text)
    return clip_vec, clap_vec


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
        return _load_audio_ffmpeg(audio_path, sample_rate=sample_rate)
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    return audio


def index_video(frames_folder: str, audio_path: str, metadata: Dict, video_id: str) -> int:
    t0 = time.perf_counter()
    frame_points = []
    audio_points = []

    # Frames (CLIP) - batched
    frame_files = [f for f in sorted(os.listdir(frames_folder)) if f.endswith(".png")]
    frame_items = []
    for file_name in frame_files:
        frame_number = int(file_name.split("_")[-1].split(".")[0])
        with Image.open(os.path.join(frames_folder, file_name)) as img:
            frame_items.append((frame_number, img.convert("RGB")))

    frame_embed_start = time.perf_counter()
    all_frame_embeddings: list[list[float]] = []
    for batch in _chunks(frame_items, max(1, settings.FRAME_BATCH_SIZE)):
        batch_images = [item[1] for item in batch]
        batch_embeddings = embed_images(batch_images)
        all_frame_embeddings.extend(batch_embeddings)
    if all_frame_embeddings:
        frame_matrix = np.asarray(all_frame_embeddings, dtype=np.float32)
        summary_vectors = max(1, settings.FRAME_SUMMARY_VECTORS)
        summary_vectors = min(summary_vectors, len(frame_matrix))
        grouped = np.array_split(frame_matrix, summary_vectors)
        for idx, group in enumerate(grouped):
            frame_mean = group.mean(axis=0)
            frame_norm = np.linalg.norm(frame_mean)
            if frame_norm > 0:
                frame_mean = frame_mean / frame_norm
            frame_points.append(
                {
                    "id": _point_id(video_id, "frame", f"summary_{idx}"),
                    "vector": {"clip": frame_mean.tolist(), "clap": [0.0] * 512},
                    "payload": {
                        "type": "frame_summary",
                        "video_id": video_id,
                        "frame_count": len(all_frame_embeddings),
                        "summary_index": idx,
                        "summary_vectors": summary_vectors,
                        **metadata,
                    },
                }
            )
    frame_embed_time = time.perf_counter() - frame_embed_start

    # Audio (CLAP) - 3-second chunks, batched
    audio_load_start = time.perf_counter()
    audio = load_audio(audio_path, sample_rate=_SAMPLE_RATE)
    audio_load_time = time.perf_counter() - audio_load_start

    chunk_size = _SAMPLE_RATE * 3
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
                    "vector": {"clip": [0.0] * 512, "clap": emb},
                    "payload": {
                        "type": "audio",
                        "video_id": video_id,
                        "timestamp": offset / _SAMPLE_RATE,
                        **metadata,
                    },
                }
            )
    if settings.AUDIO_SUMMARY_VECTOR and audio_points:
        audio_matrix = np.asarray([p["vector"]["clap"] for p in audio_points], dtype=np.float32)
        audio_mean = audio_matrix.mean(axis=0)
        audio_norm = np.linalg.norm(audio_mean)
        if audio_norm > 0:
            audio_mean = audio_mean / audio_norm
        audio_points.append(
            {
                "id": _point_id(video_id, "audio", "summary"),
                "vector": {"clip": [0.0] * 512, "clap": audio_mean.tolist()},
                "payload": {
                    "type": "audio_summary",
                    "video_id": video_id,
                    "chunk_count": len(audio_chunks),
                    **metadata,
                },
            }
        )
    audio_embed_time = time.perf_counter() - audio_embed_start

    points = frame_points + audio_points
    upsert_start = time.perf_counter()
    if points:
        _upsert_points_batched(points)
    upsert_time = time.perf_counter() - upsert_start

    total_time = time.perf_counter() - t0
    print(
        "Indexed video "
        f"{video_id} ({len(points)} points) | "
        f"frames={len(frame_points)} in {frame_embed_time:.2f}s | "
        f"audio_load={audio_load_time:.2f}s | "
        f"audio_emb={audio_embed_time:.2f}s | "
        f"upsert={upsert_time:.2f}s | total={total_time:.2f}s"
    )
    return len(points)
