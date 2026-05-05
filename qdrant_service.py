from config import settings
from embedders import embed_query
from embedders import optimize_query_parts
from embedders import parse_query
from embedders import qdrant


def search(query: str, limit: int = 20) -> list[tuple[str, float]]:
    parts = optimize_query_parts(query)
    if not parts.get("visual") and not parts.get("audio"):
        parts = parse_query(query)
    clip_vec, clap_vec = embed_query(parts)

    clip_hits = qdrant.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=clip_vec,
        using="clip",
        limit=limit,
    ).points

    clap_hits = qdrant.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=clap_vec,
        using="clap",
        limit=limit,
    ).points

    scores: dict[str, float] = {}
    for hit in clip_hits:
        vid = hit.payload["video_id"]
        scores[vid] = scores.get(vid, 0.0) + float(hit.score)

    for hit in clap_hits:
        vid = hit.payload["video_id"]
        scores[vid] = scores.get(vid, 0.0) + float(hit.score)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
