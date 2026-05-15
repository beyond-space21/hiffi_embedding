import argparse
import json
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import pika

from config import settings


def _build_list_url(limit: int, offset: int, seed: str | None) -> str:
    if not settings.BASE_API_URL:
        raise RuntimeError("BASE_API_URL is required in environment")
    base = settings.BASE_API_URL.rstrip("/")
    query = {"limit": str(limit), "offset": str(offset)}
    if seed:
        query["seed"] = seed
    return f"{base}/videos/list?{urlparse.urlencode(query)}"


def _to_absolute_video_url(video_url: str) -> str:
    if video_url.startswith("http://") or video_url.startswith("https://"):
        return video_url
    if not settings.BASE_API_VIDEO_URL:
        raise RuntimeError("BASE_API_VIDEO_URL is required to resolve relative video_url")
    return f"{settings.BASE_API_VIDEO_URL.rstrip('/')}/{video_url.lstrip('/')}"


def fetch_list_page(limit: int, offset: int, seed: str | None = None) -> tuple[list[dict], int, int, int]:
    """
    GET /videos/list for one page.

    Returns (videos, applied_limit, applied_offset, count) using API pagination
    metadata when present; otherwise falls back to request params and len(videos).
    """
    if not settings.AUTH_X_APP:
        raise RuntimeError("AUTH_X_APP is required in environment")

    list_url = _build_list_url(limit=limit, offset=offset, seed=seed)
    req = urlrequest.Request(
        list_url,
        method="GET",
        headers={
            "Auth-X-App": settings.AUTH_X_APP,
            "Accept": "application/json",
        },
    )

    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        raise RuntimeError(f"/videos/list failed with HTTP {exc.code}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"/videos/list request failed: {exc.reason}") from exc

    data = payload.get("data")
    if not isinstance(data, dict):
        data = {}

    videos = data.get("videos", [])
    if not isinstance(videos, list):
        raise RuntimeError("Unexpected /videos/list format: data.videos is not a list")

    applied_limit = data.get("limit", payload.get("limit", limit))
    applied_offset = data.get("offset", payload.get("offset", offset))
    count = data.get("count", payload.get("count", len(videos)))

    try:
        applied_limit = int(applied_limit)
    except (TypeError, ValueError):
        applied_limit = limit
    try:
        applied_offset = int(applied_offset)
    except (TypeError, ValueError):
        applied_offset = offset
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = len(videos)

    return videos, applied_limit, applied_offset, count


def build_jobs(videos: list[dict]) -> list[dict]:
    jobs = []
    for item in videos:
        video_obj = item.get("video", {})
        video_id = video_obj.get("video_id")
        video_url = video_obj.get("video_url")
        if not video_id or not video_url:
            continue
        jobs.append(
            {
                "video_id": video_id,
                "mp4_url": _to_absolute_video_url(video_url),
                "metadata": {
                    "title": video_obj.get("video_title", ""),
                    "description": video_obj.get("video_description", ""),
                    "tags": video_obj.get("video_tags", []),
                    "user_uid": video_obj.get("user_uid", ""),
                    "user_username": video_obj.get("user_username", ""),
                },
            }
        )
    return jobs


def _rabbit_channel():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=settings.RABBITMQ_HOST,
            port=settings.RABBITMQ_PORT,
            virtual_host=settings.RABBITMQ_VHOST,
            credentials=pika.PlainCredentials(
                settings.RABBITMQ_USER,
                settings.RABBITMQ_PASSWORD,
            ),
        )
    )
    return connection, connection.channel()


def publish_jobs_on_channel(channel, jobs: list[dict]) -> int:
    published = 0
    for job in jobs:
        channel.basic_publish(
            exchange="",
            routing_key=settings.RABBITMQ_QUEUE,
            body=json.dumps(job),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        published += 1
    return published


def enqueue_all_videos(
    page_size: int = 100,
    start_offset: int = 0,
    seed: str | None = None,
) -> tuple[int, int]:
    """
    Walk /videos/list with offset/limit pagination until a short page
    (count < applied_limit), then enqueue every built job.
    """
    page_size = max(1, min(100, page_size))
    offset = max(0, start_offset)
    total_published = 0
    pages = 0

    connection, channel = _rabbit_channel()
    try:
        while True:
            videos, applied_limit, applied_offset, count = fetch_list_page(
                limit=page_size,
                offset=offset,
                seed=seed,
            )
            pages += 1
            jobs = build_jobs(videos)
            total_published += publish_jobs_on_channel(channel, jobs)

            if count < applied_limit:
                break
            offset += count
    finally:
        connection.close()

    return total_published, pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enqueue all public videos from /videos/list (paginated) to RabbitMQ.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Request limit per call (1–100; API may clamp). Default 100.",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Initial offset (non-negative). Default 0.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Optional seed query parameter if the API supports it.",
    )
    args = parser.parse_args()

    total, pages = enqueue_all_videos(
        page_size=args.page_size,
        start_offset=args.start_offset,
        seed=args.seed,
    )
    print(
        f"Queued {total} video job(s) from /videos/list "
        f"({pages} page(s), page_size={max(1, min(100, args.page_size))}, "
        f"start_offset={max(0, args.start_offset)})"
    )


if __name__ == "__main__":
    main()
