import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams

from config import settings


def recreate_collection() -> None:
    client = QdrantClient(settings.QDRANT_URL)
    if client.collection_exists(settings.COLLECTION_NAME):
        client.delete_collection(collection_name=settings.COLLECTION_NAME)
    client.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config={
            "clip": VectorParams(size=512, distance=Distance.COSINE),
            "clap": VectorParams(size=512, distance=Distance.COSINE),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Flush all embeddings from Qdrant collection.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive action and flush collection.",
    )
    args = parser.parse_args()

    if not args.yes:
        print(
            "Refusing to flush without confirmation.\n"
            "Run again with: python3 flush_qdrant_embeddings.py --yes"
        )
        return

    recreate_collection()
    print(
        f"Flushed embeddings: recreated collection '{settings.COLLECTION_NAME}' "
        f"at '{settings.QDRANT_URL}'."
    )


if __name__ == "__main__":
    main()
