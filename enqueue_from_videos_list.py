import json
from urllib import parse as urlparse
from urllib import request as urlrequest
from urllib import error as urlerror

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


def fetch_videos(limit: int = 20, offset: int = 0, seed: str | None = None) -> list[dict]:
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

    videos = payload.get("data", {}).get("videos", [])
    if not isinstance(videos, list):
        raise RuntimeError("Unexpected /videos/list format: data.videos is not a list")
    return videos


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


def publish_jobs(jobs: list[dict]) -> int:
    if not jobs:
        return 0

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
    channel = connection.channel()

    published = 0
    for job in jobs:
        channel.basic_publish(
            exchange="",
            routing_key=settings.RABBITMQ_QUEUE,
            body=json.dumps(job),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        published += 1

    connection.close()
    return published


def main() -> None:
    limit = 20
    offset = 12
    seed = None

    videos = fetch_videos(limit=limit, offset=offset, seed=seed)
    jobs = build_jobs(videos)
    total = publish_jobs(jobs)
    print(f"Queued {total} video jobs from /videos/list (limit={limit}, offset={offset})")


if __name__ == "__main__":
    main()
