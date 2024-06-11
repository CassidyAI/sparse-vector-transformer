from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from pymilvus import model
from typing import Dict, List, Tuple
import time
import os

app = FastAPI()

# Set up bearer token authentication
bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("SPARSE_EMBEDDING_BEARER_TOKEN")

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return True

# Load the SPLADE embedding model
splade_ef = model.sparse.SpladeEmbeddingFunction(
    model_name="naver/splade-v3-distilbert",
    device="cpu"
)

# Define request models
class DocumentRequest(BaseModel):
    documents: List[str]

class QueryRequest(BaseModel):
    queries: List[str]

# Endpoint for embedding documents
@app.post("/embed_documents")
def embed_documents(request: DocumentRequest, authenticated: bool = Depends(validate_token)):
    start = time.time()
    docs = request.documents
    docs_embeddings = splade_ef.encode_documents(docs)
    print(f'embedded {len(docs)} documents in {(time.time() - start):.2f} seconds')
    # # convert the csr_array to list of dicts for serialization
    embeddings = []
    for doc_embedding in docs_embeddings:
        embedding_dict = {}
        for k, v in doc_embedding.todok().items():
            embedding_dict[int(k[1])] = float(v)
        embeddings.append(embedding_dict)
    return {"embeddings": embeddings}

# Endpoint for embedding queries
@app.post("/embed_queries")
def embed_queries(request: QueryRequest, authenticated: bool = Depends(validate_token)):
    start = time.time()
    queries = request.queries
    query_embeddings = splade_ef.encode_queries(queries)
    print(f'embedded {len(queries)} queries in {(time.time() - start):.2f} seconds')
    # # convert the csr_array to list of dicts for serialization
    embeddings = []
    for query_embedding in query_embeddings:
        embedding_dict = {}
        for k, v in query_embedding.todok().items():
            embedding_dict[int(k[1])] = float(v)
        embeddings.append(embedding_dict)
    return {"embeddings": embeddings}


@app.get("/health")
async def heathcheck():
    return {"status": "ok"}

class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
