from config import settings
from embedders import embed_query
from embedders import optimize_query_parts
from embedders import qdrant
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import MatchValue
import re


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _metadata_score(parts: dict, payload: dict) -> float:
    query_tokens = _tokenize(f"{parts.get('raw', '')} {parts.get('audio', '')}")
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


def _summary_payloads_by_video() -> dict[str, dict]:
    by_video: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="type", match=MatchValue(value="audio_summary"))]
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            vid = payload.get("video_id")
            if vid:
                by_video[str(vid)] = payload
        if offset is None:
            break
    return by_video


def _metadata_ranked_rows(parts: dict, limit: int) -> list[dict]:
    rows = []
    for vid, payload in _summary_payloads_by_video().items():
        meta_score = _metadata_score(parts, payload)
        if meta_score <= 0:
            continue
        rows.append(
            {
                "video_id": vid,
                "score": meta_score,
                "base_score": 0.0,
                "metadata_boost": meta_score,
                "mert_evidence": None,
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:limit]


def search(query: str, limit: int = 20) -> dict:
    parts = optimize_query_parts(query)
    _, mert_vec = embed_query(parts)

    modal_limit = max(limit * 3, settings.SEARCH_RERANK_CANDIDATES)

    mert_hits = []
    if mert_vec is not None:
        mert_hits = qdrant.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=mert_vec,
            using="mert",
            limit=modal_limit,
        ).points

    min_score = settings.SEARCH_MERT_MIN_SCORE
    mert_best: dict[str, float] = {}
    mert_evidence: dict[str, dict] = {}
    for hit in mert_hits:
        score = float(hit.score)
        if score < min_score:
            continue
        vid = hit.payload["video_id"]
        if score > mert_best.get(vid, float("-inf")):
            mert_best[vid] = score
            mert_evidence[vid] = {
                "score": score,
                "type": hit.payload.get("type"),
                "timestamp": hit.payload.get("timestamp"),
            }

    payload_by_video: dict[str, dict] = {}
    for hit in mert_hits:
        if float(hit.score) < min_score:
            continue
        vid = hit.payload["video_id"]
        if vid not in payload_by_video:
            payload_by_video[vid] = hit.payload

    all_videos = set(mert_best)
    scored_rows = []
    for vid in all_videos:
        mert_score = mert_best.get(vid, 0.0)
        base_score = mert_score
        meta_boost = settings.SEARCH_METADATA_WEIGHT * _metadata_score(parts, payload_by_video.get(vid, {}))
        final_score = base_score + meta_boost
        scored_rows.append(
            {
                "video_id": vid,
                "score": final_score,
                "base_score": base_score,
                "metadata_boost": meta_boost,
                "mert_evidence": mert_evidence.get(vid),
            }
        )

    retrieval_mode = "mert" if scored_rows else "none"
    if not scored_rows and settings.SEARCH_METADATA_FALLBACK:
        scored_rows = _metadata_ranked_rows(parts, limit)
        retrieval_mode = "metadata" if scored_rows else "none"

    scored_rows.sort(key=lambda x: x["score"], reverse=True)
    top_rows = scored_rows[:limit]
    ranked = [[row["video_id"], row["score"]] for row in top_rows]
    return {
        "optimized_query": {
            "raw": parts.get("raw", query.strip()),
            "audio": parts.get("audio", ""),
            "mert_bridge": parts.get("mert_bridge", ""),
            "has_acoustic_intent": bool(parts.get("has_acoustic_intent")),
        },
        "retrieval_mode": retrieval_mode,
        "weights": {
            "mert": 1.0 if mert_vec is not None else 0.0,
            "metadata": settings.SEARCH_METADATA_WEIGHT,
            "mert_min_score": min_score,
        },
        "evidence": top_rows,
        "results": ranked,
    }
