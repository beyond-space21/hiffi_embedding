import json
import pika
from config import settings

message = {
    "video_id": "8e136bb0294152b338b9d09646efc8f830bb8795de92901229dee7c310a8d2b2",
    "mp4_url": "https://prod.hiffi.workers.dev/videos/8e136bb0294152b338b9d09646efc8f830bb8795de92901229dee7c310a8d2b2/original.mp4",
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