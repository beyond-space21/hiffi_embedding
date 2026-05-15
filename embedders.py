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


_ACOUSTIC_INTENT_RE = re.compile(
    r"\b("
    r"music|musical|song|sound|sounds|audio|voice|voices|speak|speaking|speech|"
    r"sing|singing|vocal|vocals|beat|beats|bass|drum|drums|guitar|piano|"
    r"instrument|instruments|melody|rhythm|tempo|genre|soundtrack|ambient|"
    r"noise|noisy|quiet|silence|silent|loud|whisper|shout|laugh|applause|"
    r"crowd|explosion|gunshot|siren|rain|thunder|ocean|waves|wind|"
    r"hip[\s-]?hop|rock|jazz|pop|edm|techno|classical|orchestra|"
    r"podcast|interview|narrat|dialogue|monologue|accent|"
    r"fast|slow|quick|rapid|upbeat|energetic|energy|intense|calm|chill|"
    r"aggressive|dynamic|driving|punchy|mellow|soft|hard|heavy|light|"
    r"\[audio\]"
    r")\b",
    flags=re.I,
)

_MERT_BRIDGE_HINTS: Dict[str, str] = {
    "fast": "fast tempo, high energy, upbeat driving rhythm",
    "slow": "slow tempo, low energy, relaxed gentle rhythm",
    "quick": "fast tempo, brisk energetic rhythm",
    "rapid": "very fast tempo, intense high energy rhythm",
    "upbeat": "upbeat tempo, bright energetic major feel",
    "energetic": "high energy, fast driving rhythm",
    "calm": "slow tempo, soft low energy, peaceful tone",
    "chill": "relaxed slow tempo, mellow soft groove",
    "loud": "loud high energy, strong dynamics",
    "quiet": "quiet low volume, soft gentle tone",
}


def _expand_mert_bridge(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return ""
    return _MERT_BRIDGE_HINTS.get(normalized, text.strip())


def _truncate_bridge_text(text: str) -> str:
    text = " ".join(text.split())
    max_chars = max(32, settings.OPENAI_QUERY_BRIDGE_MAX_CHARS)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped or text[:max_chars].strip()


def _heuristic_acoustic_intent(query: str, parsed: Dict[str, str]) -> bool:
    if parsed.get("audio", "").strip():
        return True
    return _ACOUSTIC_INTENT_RE.search(query) is not None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _merge_optimized_parts(parsed: Dict[str, str], optimized: Dict[str, Any]) -> Dict[str, Any]:
    has_acoustic = _coerce_bool(optimized.get("has_acoustic_intent"), default=False)
    if not has_acoustic and _heuristic_acoustic_intent(parsed.get("raw", ""), parsed):
        has_acoustic = True

    mert_bridge = str(optimized.get("mert_bridge", "")).strip()
    audio = str(optimized.get("audio", "")).strip()
    user_audio = parsed.get("audio", "").strip()
    if user_audio:
        mert_bridge = user_audio if not mert_bridge else f"{user_audio}. {mert_bridge}"
        audio = user_audio if not audio else f"{user_audio}. {audio}"

    if not mert_bridge and audio:
        mert_bridge = audio
    if not audio and mert_bridge:
        audio = mert_bridge

    if has_acoustic:
        mert_bridge = _truncate_bridge_text(_expand_mert_bridge(mert_bridge)) if mert_bridge else ""
        if not mert_bridge:
            fallback = parsed.get("audio") or parsed.get("raw", "")
            mert_bridge = _truncate_bridge_text(_expand_mert_bridge(fallback))
            if not audio:
                audio = fallback.strip()
    else:
        mert_bridge = ""
        audio = audio or parsed.get("audio", "")

    return {
        "visual": parsed.get("visual", ""),
        "audio": audio,
        "mert_bridge": mert_bridge,
        "has_acoustic_intent": has_acoustic and bool(mert_bridge),
        "raw": parsed.get("raw", ""),
    }


def _synthesize_query_audio(query_text: str) -> np.ndarray | None:
    if not settings.OPENAI_AUDIO_BRIDGE_ENABLED:
        return None
    text = _truncate_bridge_text(query_text.strip())
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


def optimize_query_parts(query: str) -> Dict[str, Any]:
    parsed = parse_query(query)
    if not settings.OPENAI_QUERY_OPTIMIZATION:
        has_acoustic = _heuristic_acoustic_intent(query, parsed)
        audio = parsed.get("audio", "").strip()
        if has_acoustic and not audio:
            audio = parsed.get("raw", "").strip()
        mert_bridge = _truncate_bridge_text(_expand_mert_bridge(audio)) if has_acoustic else ""
        return {
            "visual": parsed.get("visual", ""),
            "audio": audio,
            "mert_bridge": mert_bridge,
            "has_acoustic_intent": has_acoustic and bool(mert_bridge),
            "raw": parsed.get("raw", query.strip()),
        }

    system_prompt = (
        "You rewrite video-search queries for MERT acoustic retrieval.\n"
        "Pipeline: mert_bridge is spoken by TTS, embedded by MERT (trained on real music/audio waveforms), "
        "then matched against 3-second soundtrack chunks from videos.\n"
        "Return JSON only with keys: has_acoustic_intent, mert_bridge, audio.\n"
        "- has_acoustic_intent (boolean): true when the user wants sound/music/speech/ambience "
        "or describes pace/energy/tempo (e.g. fast, slow, upbeat, calm); "
        "false only for purely visual/object/topic queries with no sonic or pace cue.\n"
        "- mert_bridge (string, <=100 chars): short English phrase for TTS. Use concrete sonic vocabulary "
        "(instruments, genre, tempo, energy, vocal tone, loudness, rhythm, room tone). "
        "Describe how it should SOUND, not what is seen. No dialogue quotes or lyrics.\n"
        "- audio (string): same intent as mert_bridge, may be slightly longer; used for metadata matching.\n"
        "Rules:\n"
        "- Never invent background audio the user did not imply.\n"
        "- Do not translate visual scenes into guessed ambience unless the user mentions sound/audio/music.\n"
        "- For music: genre, instruments, tempo, mood — not song titles or artist names alone.\n"
        "- For speech: pace, energy, gender/tone — not what is being said.\n"
        "Examples:\n"
        '{"has_acoustic_intent":false,"mert_bridge":"","audio":""} for "red sports car chase"\n'
        '{"has_acoustic_intent":true,"mert_bridge":"slow jazz piano, brushed drums, warm mellow","audio":"slow jazz piano with brushed drums"} for "jazz piano background"\n'
        '{"has_acoustic_intent":true,"mert_bridge":"calm clear female speech, slow steady pace","audio":"calm female narration"} for "calm female narrator"\n'
        '{"has_acoustic_intent":true,"mert_bridge":"fast tempo, high energy, upbeat driving rhythm","audio":"fast energetic"} for "fast"\n'
        "Do not include markdown."
    )

    try:
        response = openai_client.chat.completions.create(
            model=settings.OPENAI_QUERY_MODEL,
            temperature=0.0,
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
        return _merge_optimized_parts(parsed, optimized)
    except Exception as exc:
        print(f"OpenAI query optimization failed, using raw query: {exc}")
        has_acoustic = _heuristic_acoustic_intent(query, parsed)
        audio = parsed.get("audio", "").strip() or (parsed.get("raw", "").strip() if has_acoustic else "")
        mert_bridge = _truncate_bridge_text(_expand_mert_bridge(audio)) if has_acoustic else ""
        return {
            "visual": parsed.get("visual", ""),
            "audio": audio,
            "mert_bridge": mert_bridge,
            "has_acoustic_intent": has_acoustic and bool(mert_bridge),
            "raw": parsed.get("raw", query.strip()),
        }


def embed_query(parts: Dict[str, Any]) -> Tuple[list[float], list[float] | None]:
    if not parts.get("has_acoustic_intent"):
        return [], None
    bridge_text = parts.get("mert_bridge") or parts.get("audio") or ""
    bridge_audio = _synthesize_query_audio(bridge_text)
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
