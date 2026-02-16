from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import hashlib

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str


# Deterministic fake embedding (so same text → same vector)
def fake_embedding(text: str):
    hash_object = hashlib.sha256(text.encode())
    seed = int(hash_object.hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.rand(384)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):

    query_embedding = fake_embedding(request.query)

    similarities = []

    for doc in request.docs:
        doc_embedding = fake_embedding(doc)
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((sim, doc))

    similarities.sort(reverse=True, key=lambda x: x[0])
    top_matches = [doc for _, doc in similarities[:3]]

    return {"matches": top_matches}