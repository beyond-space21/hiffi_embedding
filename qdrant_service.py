from config import settings
from embedders import embed_query
from embedders import optimize_query_parts
from embedders import parse_query
from embedders import qdrant
import re


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _infer_query_intent(parts: dict) -> tuple[float, float]:
    if not settings.SEARCH_ENABLE_INTENT_WEIGHTS:
        return settings.SEARCH_CLIP_WEIGHT, settings.SEARCH_CLAP_WEIGHT

    text = f"{parts.get('raw', '')} {parts.get('visual', '')} {parts.get('audio', '')}".lower()
    audio_cues = {
        "sound", "audio", "voice", "music", "song", "noise", "engine", "speech",
        "talking", "singing", "applause", "laugh", "scream", "horn",
    }
    visual_cues = {
        "car", "person", "man", "woman", "dog", "cat", "street", "beach",
        "object", "color", "scene", "view", "camera", "dance", "running",
    }
    audio_score = sum(1 for w in audio_cues if w in text)
    visual_score = sum(1 for w in visual_cues if w in text)
    if audio_score == 0 and visual_score == 0:
        return settings.SEARCH_CLIP_WEIGHT, settings.SEARCH_CLAP_WEIGHT

    total = audio_score + visual_score
    # Keep a floor so either modality can still contribute.
    clip_w = max(0.1, min(0.9, visual_score / total if total else settings.SEARCH_CLIP_WEIGHT))
    clap_w = 1.0 - clip_w
    return clip_w, clap_w


def _metadata_score(parts: dict, payload: dict) -> float:
    query_tokens = _tokenize(
        f"{parts.get('raw', '')} {parts.get('visual', '')} {parts.get('audio', '')}"
    )
    if not query_tokens:
        return 0.0
    metadata_text = " ".join(
        [
            str(payload.get("title", "")),
            str(payload.get("description", "")),
            " ".join(payload.get("tags", []) if isinstance(payload.get("tags"), list) else []),
        ]
    )
    meta_tokens = _tokenize(metadata_text)
    if not meta_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(meta_tokens))
    return overlap / max(len(query_tokens), 1)


def search(query: str, limit: int = 20) -> dict:
    parts = optimize_query_parts(query)
    if not parts.get("visual") and not parts.get("audio"):
        parts = parse_query(query)
    clip_vec, clap_vec = embed_query(parts)

    modal_limit = max(limit * 3, settings.SEARCH_RERANK_CANDIDATES)

    clip_hits = qdrant.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=clip_vec,
        using="clip",
        limit=modal_limit,
    ).points

    clap_hits = qdrant.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=clap_vec,
        using="clap",
        limit=modal_limit,
    ).points

    clip_best: dict[str, float] = {}
    clip_evidence: dict[str, dict] = {}
    for hit in clip_hits:
        vid = hit.payload["video_id"]
        score = float(hit.score)
        if score > clip_best.get(vid, float("-inf")):
            clip_best[vid] = score
            clip_evidence[vid] = {
                "score": score,
                "type": hit.payload.get("type"),
                "summary_index": hit.payload.get("summary_index"),
            }

    clap_best: dict[str, float] = {}
    clap_evidence: dict[str, dict] = {}
    for hit in clap_hits:
        vid = hit.payload["video_id"]
        score = float(hit.score)
        if score > clap_best.get(vid, float("-inf")):
            clap_best[vid] = score
            clap_evidence[vid] = {
                "score": score,
                "type": hit.payload.get("type"),
                "timestamp": hit.payload.get("timestamp"),
            }

    clip_weight, clap_weight = _infer_query_intent(parts)

    payload_by_video: dict[str, dict] = {}
    for hit in clip_hits:
        vid = hit.payload["video_id"]
        if vid not in payload_by_video:
            payload_by_video[vid] = hit.payload
    for hit in clap_hits:
        vid = hit.payload["video_id"]
        if vid not in payload_by_video:
            payload_by_video[vid] = hit.payload

    all_videos = set(clip_best) | set(clap_best)
    scored_rows = []
    for vid in all_videos:
        clip_score = clip_best.get(vid, 0.0)
        clap_score = clap_best.get(vid, 0.0)
        base_score = (clip_weight * clip_score) + (clap_weight * clap_score)
        meta_boost = settings.SEARCH_METADATA_WEIGHT * _metadata_score(parts, payload_by_video.get(vid, {}))
        final_score = base_score + meta_boost
        scored_rows.append(
            {
                "video_id": vid,
                "score": final_score,
                "base_score": base_score,
                "metadata_boost": meta_boost,
                "clip_evidence": clip_evidence.get(vid),
                "clap_evidence": clap_evidence.get(vid),
            }
        )

    scored_rows.sort(key=lambda x: x["score"], reverse=True)
    top_rows = scored_rows[:limit]
    ranked = [[row["video_id"], row["score"]] for row in top_rows]
    return {
        "optimized_query": {
            "raw": parts.get("raw", query.strip()),
            "visual": parts.get("visual", ""),
            "audio": parts.get("audio", ""),
        },
        "weights": {
            "clip": clip_weight,
            "clap": clap_weight,
            "metadata": settings.SEARCH_METADATA_WEIGHT,
        },
        "evidence": top_rows,
        "results": ranked,
    }
