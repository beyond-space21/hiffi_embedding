from config import settings
from embedders import embed_query
from embedders import optimize_query_parts
from embedders import parse_query
from embedders import qdrant


def search(query: str, limit: int = 20) -> dict:
    parts = optimize_query_parts(query)
    if not parts.get("visual") and not parts.get("audio"):
        parts = parse_query(query)
    clip_vec, clap_vec = embed_query(parts)

    modal_limit = max(limit * 3, 50)

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
    for hit in clip_hits:
        vid = hit.payload["video_id"]
        score = float(hit.score)
        clip_best[vid] = max(clip_best.get(vid, float("-inf")), score)

    clap_best: dict[str, float] = {}
    for hit in clap_hits:
        vid = hit.payload["video_id"]
        score = float(hit.score)
        clap_best[vid] = max(clap_best.get(vid, float("-inf")), score)

    all_videos = set(clip_best) | set(clap_best)
    scores: dict[str, float] = {}
    for vid in all_videos:
        clip_score = clip_best.get(vid, 0.0)
        clap_score = clap_best.get(vid, 0.0)
        scores[vid] = (settings.SEARCH_CLIP_WEIGHT * clip_score) + (
            settings.SEARCH_CLAP_WEIGHT * clap_score
        )

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {
        "optimized_query": {
            "raw": parts.get("raw", query.strip()),
            "visual": parts.get("visual", ""),
            "audio": parts.get("audio", ""),
        },
        "weights": {
            "clip": settings.SEARCH_CLIP_WEIGHT,
            "clap": settings.SEARCH_CLAP_WEIGHT,
        },
        "results": ranked,
    }
