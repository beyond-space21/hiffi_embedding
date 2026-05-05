import json
import pika
from config import settings

message = {
    "video_id": "724c3d2b22ad57a7ce84841327958a6b92778f2d0e50f45a2eb5b6d08072ea40",
    "mp4_url": "https://dev.hiffi.workers.dev/videos/724c3d2b22ad57a7ce84841327958a6b92778f2d0e50f45a2eb5b6d08072ea40/original.mp4",
    "metadata": {"title": "Car jumping on street", "source": "youtube"},
}

connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        virtual_host=settings.RABBITMQ_VHOST,
        credentials=pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASSWORD),
    )
)
channel = connection.channel()
channel.basic_publish(
    exchange="",
    routing_key=settings.RABBITMQ_QUEUE,
    body=json.dumps(message),
    properties=pika.BasicProperties(delivery_mode=2),
)
connection.close()
print("Job sent")