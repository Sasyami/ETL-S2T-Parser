import os
import json
import logging
import numpy as np
from typing import List, Dict, Any
from langchain_gigachat.embeddings import GigaChatEmbeddings
from db_storage import get_db_connection, generate_id
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_API_KEY") or os.getenv("GIGACHAT_EMBEDDINGS_CREDENTIALS")
if not GIGACHAT_CREDENTIALS:
    raise ValueError("Missing GigaChat credentials")

GIGACHAT_BASE_URL = os.getenv("GIGACHAT_API_URL", "https://gigachat.devices.sberbank.ru/api/v1")
VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"
SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
MODEL = os.getenv("MODEL", "GigaChat-Pro")
TIMEOUT = int(os.getenv("GIGACHAT_TIMEOUT", "120"))

# Initialize GigaChat Embeddings
try:
    embeddings_model = GigaChatEmbeddings(
        credentials=GIGACHAT_CREDENTIALS,  # will read from env GIGACHAT_CREDENTIALS
        verify_ssl_certs=VERIFY_SSL,
        scope=SCOPE,
        model="EmbeddingsGigaR",
    )
    logger.info("GigaChat Embeddings initialized")
except Exception as e:
    logger.error(f"Failed to init embeddings: {e}")
    raise

def create_embedding(text: str) -> str:
    """Generate embedding vector as JSON string."""
    vec = embeddings_model.embed_query(text)
    return json.dumps(vec)

def store_embedding(entity_id: str, entity_type: str, text: str):
    """Store embedding for an entity."""
    if not text:
        return
    vec_json = create_embedding(text)
    conn = get_db_connection()
    cursor = conn.cursor()
    emb_id = generate_id(entity_id, entity_type, text[:50])
    cursor.execute("""
        INSERT OR REPLACE INTO embeddings (id, entity_id, entity_type, vector)
        VALUES (?, ?, ?, ?)
    """, (emb_id, entity_id, entity_type, vec_json))
    conn.commit()
    conn.close()

def similarity_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Find entities with most similar embeddings."""
    query_vec = np.array(json.loads(create_embedding(query)))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT entity_id, entity_type, vector FROM embeddings")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return []
    similarities = []
    qn = np.linalg.norm(query_vec)
    if qn == 0:
        return []
    for row in rows:
        vec = np.array(json.loads(row["vector"]))
        vn = np.linalg.norm(vec)
        if vn == 0:
            continue
        sim = float(np.dot(query_vec, vec) / (qn * vn))
        similarities.append((row["entity_id"], row["entity_type"], sim))
    similarities.sort(key=lambda x: x[2], reverse=True)
    return [
        {"entity_id": eid, "entity_type": typ, "similarity": float(sim)}
        for eid, typ, sim in similarities[:top_k]
    ]

def find_similar_columns(name: str) -> List[Dict[str, Any]]:
    """Convenience to find similar columns."""
    return similarity_search(name)