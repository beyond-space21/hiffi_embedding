from config import settings
from embedders import embed_query
from embedders import optimize_query_parts
from embedders import qdrant
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

    mert_best: dict[str, float] = {}
    mert_evidence: dict[str, dict] = {}
    for hit in mert_hits:
        vid = hit.payload["video_id"]
        score = float(hit.score)
        if score > mert_best.get(vid, float("-inf")):
            mert_best[vid] = score
            mert_evidence[vid] = {
                "score": score,
                "type": hit.payload.get("type"),
                "timestamp": hit.payload.get("timestamp"),
            }

    payload_by_video: dict[str, dict] = {}
    for hit in mert_hits:
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

    scored_rows.sort(key=lambda x: x["score"], reverse=True)
    top_rows = scored_rows[:limit]
    ranked = [[row["video_id"], row["score"]] for row in top_rows]
    return {
        "optimized_query": {
            "raw": parts.get("raw", query.strip()),
            "audio": parts.get("audio", ""),
        },
        "weights": {
            "mert": 1.0 if mert_vec is not None else 0.0,
            "metadata": settings.SEARCH_METADATA_WEIGHT,
        },
        "evidence": top_rows,
        "results": ranked,
    }
