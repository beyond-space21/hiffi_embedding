import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from qdrant_service import search
from embedders import runtime_device_label

app = FastAPI(title="Video Semantic Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


@app.post("/search")
async def search_endpoint(req: SearchRequest) -> dict:
    results = search(req.query, req.limit)
    return {"results": results}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "device": runtime_device_label()}


if __name__ == "__main__":
    import uvicorn

    print(f"Embedding runtime device: {runtime_device_label()}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
