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

    # Create embedding for query
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.query
    )
    query_embedding = query_response.data[0].embedding

    # Create embeddings for all documents
    docs_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.docs
    )

    similarities = []

    for i, doc_obj in enumerate(docs_response.data):
        doc_embedding = doc_obj.embedding
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, sim))  # store index + similarity

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Extract top 3 indexes
    top_matches = [idx for idx, _ in similarities[:3]]

    return {"matches": top_matches}