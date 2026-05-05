import json
import pika
from config import settings

message = {
    "video_id": "4dcc721e0150fe09dda2a2a7fadb24d75bca545609c291809e57fe44bb99f414",
    "mp4_url": "https://dev.hiffi.workers.dev/videos/4dcc721e0150fe09dda2a2a7fadb24d75bca545609c291809e57fe44bb99f414/original.mp4",
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