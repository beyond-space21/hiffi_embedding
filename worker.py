import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib import error as urlerror
from urllib import request as urlrequest

import pika

from config import settings


def _safe_video_dir(video_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in video_id)


def preprocess_video(mp4_url: str, video_id: str) -> tuple[str, str]:
    safe_video_id = _safe_video_dir(video_id)
    base_dir = Path(settings.TEMP_DIR) / safe_video_id
    frames_dir = base_dir / "frames"
    audio_path = base_dir / "audio.aac"

    if base_dir.exists():
        shutil.rmtree(base_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "make",
                "-s",
                "preprocess-video",
                f"TEMP_DIR={settings.TEMP_DIR}",
                f"VIDEO_ID={safe_video_id}",
                f"VIDEO_URL={mp4_url}",
                f"FRAME_EXTRACT_FPS={settings.FRAME_EXTRACT_FPS}",
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("make/ffmpeg is not installed or not available in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"make preprocessing failed: {exc}") from exc

    if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
        raise RuntimeError(f"no frames were generated for '{video_id}'")
    if not audio_path.exists():
        raise RuntimeError(f"audio file was not generated for '{video_id}'")

    return str(frames_dir), str(audio_path)


def consume_forever(worker_id: int) -> None:
    # Import inside child process to avoid CUDA initialization in parent.
    from embedders import ensure_collection
    from embedders import index_video
    from embedders import runtime_device_label

    def post_vector_position(video_id: str, vector_position: int) -> None:
        if not settings.BASE_API_URL or not settings.AUTH_X_APP:
            raise RuntimeError("BASE_API_URL and AUTH_X_APP must be set in environment")

        base_url = settings.BASE_API_URL.rstrip("/")
        url = f"{base_url}/videos/embedding/vector-position/{video_id}"
        payload = json.dumps({"vector_position": vector_position}).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Auth-X-App": settings.AUTH_X_APP,
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=20) as resp:
                status_code = getattr(resp, "status", None) or resp.getcode()
            if status_code < 200 or status_code >= 300:
                raise RuntimeError(f"vector position API returned status {status_code}")
        except urlerror.HTTPError as exc:
            raise RuntimeError(f"vector position API HTTP {exc.code}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"vector position API request failed: {exc.reason}") from exc

    def callback(ch, method, properties, body) -> None:
        msg = None
        try:
            msg = json.loads(body)
            video_id = msg["video_id"]
            mp4_url = msg["mp4_url"]
            print(
                f"[worker-{worker_id}] Processing video: {video_id} "
                f"(frame_fps={settings.FRAME_EXTRACT_FPS})"
            )

            t0 = time.perf_counter()
            preprocess_start = time.perf_counter()
            frames_folder, audio_path = preprocess_video(mp4_url=mp4_url, video_id=video_id)
            preprocess_time = time.perf_counter() - preprocess_start

            index_start = time.perf_counter()
            vector_position = index_video(
                frames_folder=frames_folder,
                audio_path=audio_path,
                metadata=msg.get("metadata", {}),
                video_id=video_id,
            )
            post_start = time.perf_counter()
            post_vector_position(video_id=video_id, vector_position=vector_position)
            post_time = time.perf_counter() - post_start
            index_time = time.perf_counter() - index_start
            total_time = time.perf_counter() - t0

            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(
                f"[worker-{worker_id}] Finished {video_id} | "
                f"preprocess={preprocess_time:.2f}s | "
                f"index={index_time:.2f}s | "
                f"notify={post_time:.2f}s | "
                f"job_total={total_time:.2f}s"
            )

        except Exception as e:
            failed_video_id = "unknown"
            if isinstance(msg, dict):
                failed_video_id = msg.get("video_id", "unknown")
            print(f"[worker-{worker_id}] Failed {failed_video_id}: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    print(f"[worker-{worker_id}] Embedding runtime device: {runtime_device_label()}")
    ensure_collection()

    credentials = pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        virtual_host=settings.RABBITMQ_VHOST,
        credentials=credentials,
        heartbeat=600,
    )

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(
        queue=settings.RABBITMQ_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": settings.RABBITMQ_DLQ,
        },
    )
    channel.queue_declare(queue=settings.RABBITMQ_DLQ, durable=True)

    channel.basic_qos(prefetch_count=max(1, settings.RABBITMQ_PREFETCH_COUNT))
    channel.basic_consume(queue=settings.RABBITMQ_QUEUE, on_message_callback=callback)

    print(
        f"[worker-{worker_id} pid={os.getpid()}] listening on "
        f"'{settings.RABBITMQ_QUEUE}' (DLQ enabled, prefetch={max(1, settings.RABBITMQ_PREFETCH_COUNT)})"
    )
    channel.start_consuming()


def main() -> None:
    process_count = max(1, settings.WORKER_PROCESSES)
    print(f"Starting {process_count} worker process(es)")
    mp.set_start_method("spawn", force=True)

    if process_count == 1:
        consume_forever(worker_id=1)
        return

    processes = []
    for i in range(process_count):
        p = mp.Process(target=consume_forever, args=(i + 1,), daemon=False)
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping worker processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
