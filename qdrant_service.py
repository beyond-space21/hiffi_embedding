from config import settings
from embedders import embed_query
from embedders import openai_client
from embedders import parse_query
from embedders import qdrant


def search(query: str, limit: int = 20) -> list[tuple[str, float]]:
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


def _build_result_summary(results: list[tuple[str, float]], max_items: int = 10) -> str:
    if not results:
        return "No matching videos found."
    lines = []
    for idx, (video_id, score) in enumerate(results[:max_items], start=1):
        lines.append(f"{idx}. video_id={video_id}, score={score:.4f}")
    return "\n".join(lines)


def generate_openai_response(query: str, results: list[tuple[str, float]]) -> str:
    context = _build_result_summary(results)
    prompt = (
        "You are a video search assistant.\n"
        "Given a user query and ranked video results, produce a concise response:\n"
        "- One short summary sentence.\n"
        "- Then top recommendations in bullets using video_id.\n\n"
        f"User query: {query}\n\n"
        f"Ranked results:\n{context}"
    )
    try:
        response = openai_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return (response.output_text or "").strip()
    except Exception as exc:
        return f"OpenAI response unavailable: {exc}"


def search_with_openai(query: str, limit: int = 20) -> dict:
    results = search(query=query, limit=limit)
    ai_response = generate_openai_response(query=query, results=results)
    return {"results": results, "openai_response": ai_response}
