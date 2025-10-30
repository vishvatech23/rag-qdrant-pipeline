import os
import re
import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import fitz  # pymupdf
import docx
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import nltk
from tqdm import tqdm

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# ----------------- Configuration -----------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
DEFAULT_COLLECTION = "rag_chunks"

QDRANT_HOST = os.getenv('QDRANT_HOST', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# ----------------- Utilities -----------------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def now_ts() -> int:
    """Return current UTC epoch seconds for numeric range queries."""
    return int(datetime.utcnow().timestamp())

def file_id(path: str) -> str:
    h = hashlib.sha1(path.encode('utf-8')).hexdigest()
    return f"{Path(path).stem}_{h[:10]}"

# ----------------- Ingestors -----------------
def read_pdf(path: str) -> Tuple[str, Dict]:
    doc = fitz.open(path)
    pages = []
    for p in doc:
        pages.append(p.get_text("text") or "")
    full = "\n".join(pages)
    return full, {"pages": len(pages)}

def read_docx(path: str) -> Tuple[str, Dict]:
    doc = docx.Document(path)
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras), {}

def read_html(path: str) -> Tuple[str, Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    for s in soup(['script', 'style']):
        s.decompose()
    text = soup.get_text(separator='\n')
    return text, {}

# ----------------- Normalization -----------------
def normalize_text(text: str) -> str:
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 20:
        freq = {}
        for l in lines[:20]:
            freq[l] = freq.get(l, 0) + 1
        repeats = {k for k, v in freq.items() if v >= 2}
        if repeats:
            text = '\n'.join([l for l in lines if l not in repeats])
    return text

# ----------------- Chunking -----------------
def heading_based_chunks(text: str, max_chars: int = 2000) -> List[Dict]:
    lines = text.splitlines()
    chunks = []
    current = {"heading": None, "text": []}
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if (s.isupper() and len(s) < 120) or s.lower().startswith('chapter') or s.endswith(':'):
            if current['text']:
                chunks.append(current)
            current = {"heading": s, "text": []}
        else:
            current['text'].append(s)
            if len(" ".join(current['text'])) >= max_chars:
                chunks.append(current)
                current = {"heading": None, "text": []}
    if current['text']:
        chunks.append(current)
    out = []
    for c in chunks:
        txt = ' '.join(c['text']).strip()
        if txt:
            out.append({'heading': c['heading'], 'text': txt})
    return out

def semantic_sliding_chunks(text: str, target_chars: int = 1500, overlap_chars: int = 300) -> List[Dict]:
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        sl = len(s)
        if cur_len + sl > target_chars and cur:
            chunks.append({'heading': None, 'text': ' '.join(cur)})
            # build overlap
            keep = []
            klen = 0
            for sent in reversed(cur):
                keep.insert(0, sent)
                klen += len(sent)
                if klen >= overlap_chars:
                    break
            cur = keep.copy()
            cur_len = sum(len(x) for x in cur)
        cur.append(s)
        cur_len += sl
    if cur:
        chunks.append({'heading': None, 'text': ' '.join(cur)})
    return chunks

def chunk_document(text: str) -> List[Dict]:
    text = normalize_text(text)
    hb = heading_based_chunks(text, max_chars=2000)
    if len(hb) >= 3:
        final = []
        for part in hb:
            if len(part['text']) > 3000:
                final.extend(semantic_sliding_chunks(part['text']))
            else:
                final.append(part)
        return final
    return semantic_sliding_chunks(text)

# ----------------- RAG Indexer -----------------
class RAGIndexer:
    def __init__(
        self,
        qdrant_host: str = QDRANT_HOST,
        api_key: Optional[str] = QDRANT_API_KEY,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = EMBEDDING_MODEL,
        dim: int = EMBED_DIM
    ):
        self.collection_name = collection_name
        self._embedding_model_name = embedding_model
        self.dim = dim
        # lazy load model to avoid heavy work at import-time if desired
        self.model: Optional[SentenceTransformer] = None
        self._ensure_model_loaded()

        # connect to Qdrant
        self.qclient = QdrantClient(url=qdrant_host, api_key=api_key) if api_key else QdrantClient(url=qdrant_host)
        print(f"🔗 Connecting to Qdrant at {qdrant_host}")
        self._ensure_collection()
        print(f"✅ RAGIndexer initialized with collection '{collection_name}'")

    def _ensure_model_loaded(self):
        if self.model is None:
            print(f"🧠 Loading embedding model: {self._embedding_model_name} (this may take a moment)")
            self.model = SentenceTransformer(self._embedding_model_name)

    # ---------- simple token count ----------
    @staticmethod
    def _count_tokens(text: str) -> int:
        # approximate tokens by splitting on whitespace; good enough for chunk sizing
        return len([t for t in re.split(r"\s+", text.strip()) if t])

    # ---------- chunking (token-based) ----------
    def chunk_text(self, text: str, metadata: Optional[Dict] = None, min_tokens: int = 300, max_tokens: int = 500, overlap_tokens: int = 50) -> List[Dict]:
        text = normalize_text(text)
        sentences = sent_tokenize(text)
        chunks: List[Dict] = []
        current_sentences = []
        current_tokens = 0

        def flush_chunk(force: bool = False):
            nonlocal current_sentences, current_tokens
            if not current_sentences:
                return
            tokens_in_chunk = self._count_tokens(" ".join(current_sentences))
            if force or tokens_in_chunk >= min_tokens or tokens_in_chunk > 0:
                chunk_text_val = " ".join(current_sentences).strip()
                if chunk_text_val:
                    item = {'heading': None, 'text': chunk_text_val}
                    if metadata:
                        item['metadata'] = dict(metadata)
                    chunks.append(item)
                current_sentences = []
                current_tokens = 0

        for s in sentences:
            stokens = self._count_tokens(s)
            if current_tokens + stokens > max_tokens and current_sentences:
                # compute overlap
                overlap_list = []
                overlap_count = 0
                for sent in reversed(current_sentences):
                    overlap_list.insert(0, sent)
                    overlap_count += self._count_tokens(sent)
                    if overlap_count >= overlap_tokens:
                        break
                flush_chunk(force=True)
                current_sentences = overlap_list.copy()
                current_tokens = sum(self._count_tokens(x) for x in current_sentences)
            current_sentences.append(s)
            current_tokens += stokens

        flush_chunk(force=True)
        return chunks

    # ---------- Qdrant collection management ----------
    def _ensure_collection(self):
        try:
            collections = self.qclient.get_collections().collections
        except Exception:
            collections = []
        names = [c.name for c in collections]
        if self.collection_name in names:
            print(f"Collection '{self.collection_name}' already exists.")
            return

        print(f"Creating collection '{self.collection_name}' (dim={self.dim})")
        # NOTE: keep only supported kwargs (do NOT pass shards_count)
        try:
            self.qclient.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=self.dim, distance=rest.Distance.COSINE),
                replication_factor=1
            )
        except TypeError:
            # fallback: create_collection if recreate_collection signature differs
            try:
                self.qclient.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(size=self.dim, distance=rest.Distance.COSINE)
                )
            except Exception as e:
                print("⚠️ Failed to create collection:", e)
                raise

    # ---------- metadata creation ----------
    def _make_meta(self, doc_id: str, source: str, chunk_id: str, heading: Optional[str], page_numbers: Optional[List[int]] = None, tags: Optional[List[str]] = None) -> Dict:
        return {
            'doc_id': doc_id,
            'source': source,
            'chunk_id': chunk_id,
            'heading': heading,
            'page_numbers': page_numbers or [],
            'created_at': now_iso(),
            'created_at_ts': now_ts(),  # numeric epoch for reliable range queries
            'tags': tags or [],
            'embedding_model': self._embedding_model_name,
            'text': None  # populated later
        }

    # ---------- ingestion ----------
    def ingest_file(self, path: str) -> int:
        p = Path(path)
        ext = p.suffix.lower()
        if ext == '.pdf':
            raw, meta = read_pdf(path)
            page_numbers = list(range(1, meta.get('pages', 0) + 1))
        elif ext == '.docx':
            raw, meta = read_docx(path)
            page_numbers = []
        elif ext in ('.html', '.htm'):
            raw, meta = read_html(path)
            page_numbers = []
        elif ext == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                raw = f.read()
            page_numbers = []
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docid = file_id(str(p))
        base_meta = {
            'doc_id': docid,
            'source': str(p),
            'page_numbers': page_numbers,
        }
        chunks = self.chunk_text(raw, metadata=base_meta, min_tokens=300, max_tokens=500, overlap_tokens=50)
        texts = [c['text'] for c in chunks]
        if not texts:
            print("No chunks produced.")
            return 0

        embeddings = self.embed_chunks(texts)
        points = []
        for i, (c, emb) in enumerate(zip(chunks, embeddings)):
            cid = f"{docid}::chunk_{i}"
            meta = self._make_meta(docid, str(p), cid, c.get('heading'), page_numbers, tags=self._detect_tags(c['text']))
            # merge provided metadata if present (safe merge)
            if 'metadata' in c and isinstance(c['metadata'], dict):
                for k, v in c['metadata'].items():
                    if k not in meta:
                        meta[k] = v
            meta['text'] = c['text'][:4000]
            # build PointStruct
            point = rest.PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload=meta)
            points.append(point)

        try:
            self.store_in_qdrant(points)
        except Exception as e:
            print("⚠️ Error while upserting to Qdrant:", e)
            raise
        return len(points)

    # ---------- embeddings ----------
    def embed_chunks(self, texts: List[str]):
        self._ensure_model_loaded()
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        import numpy as np
        from numpy.linalg import norm
        norms = norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def store_in_qdrant(self, points: List[rest.PointStruct], batch_size: int = 64) -> None:
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.qclient.upsert(collection_name=self.collection_name, points=batch)
            except Exception as e:
                print(f"⚠️ Qdrant upsert failed for batch starting at {i}: {e}")
                raise

    # ---------- tag heuristics ----------
    def _detect_tags(self, text: str) -> List[str]:
        tags = []
        if 'http://' in text or 'https://' in text:
            tags.append('link')
        if re.search(r'\b\w+@\w+\.', text):
            tags.append('email')
        if re.search(r'\bdef\s+\w+\(|\bclass\s+\w+\(|;\n|\{\n|\}\n', text):
            tags.append('code')
        if '|' in text and '\n' in text:
            tags.append('table')
        return tags

    # ---------- filter builder ----------
    def _build_filter(self, doc_id: Optional[str] = None, tags: Optional[List[str]] = None, date_from_ts: Optional[int] = None, date_to_ts: Optional[int] = None):
        must = []
        if doc_id:
            must.append(rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=doc_id)))
        if tags:
            must.append(rest.FieldCondition(key="tags", match=rest.MatchAny(any=tags)))
        if date_from_ts is not None or date_to_ts is not None:
            rng = {}
            if date_from_ts is not None:
                rng['gte'] = date_from_ts
            if date_to_ts is not None:
                rng['lte'] = date_to_ts
            must.append(rest.FieldCondition(key="created_at_ts", range=rest.Range(**rng)))
        if not must:
            return None
        return rest.Filter(must=must)

    # ---------- query ----------
    def query(self, query_text: str, top_k: int = 5, filter_payload: Optional[rest.Filter] = None, *, doc_id: Optional[str] = None, tags: Optional[List[str]] = None, date_from_ts: Optional[int] = None, date_to_ts: Optional[int] = None) -> List[Dict]:
        print(f"🔍 Querying: '{query_text[:80]}' (top_k={top_k})")
        self._ensure_model_loaded()
        q_emb = self.model.encode([query_text], convert_to_numpy=True)
        import numpy as np
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

        q_filter = filter_payload if filter_payload is not None else self._build_filter(doc_id=doc_id, tags=tags, date_from_ts=date_from_ts, date_to_ts=date_to_ts)
        search_result = self.qclient.search(collection_name=self.collection_name, query_vector=q_emb[0].tolist(), limit=top_k, with_payload=True, query_filter=q_filter)

        out = []
        for r in search_result:
            payload = r.payload or {}
            out.append({
                'score': float(r.score),
                'text': payload.get('text', ''),
                'payload': payload
            })
        print(f"✅ Found {len(out)} results")
        return out

    # ---------- convenience wrappers ----------
    def normalize_text(self, text: str) -> str:
        return normalize_text(text)

    def semantic_chunk(self, text: str, max_tokens: int = 400, overlap: int = 50) -> List[Dict]:
        return self.chunk_text(text, min_tokens=max_tokens // 2, max_tokens=max_tokens, overlap_tokens=overlap)

    def add_metadata(self, chunks: List[Dict], doc_id: str, source: str, page_numbers: Optional[List[int]] = None) -> List[Dict]:
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}::chunk_{i}"
            chunk['metadata'] = {
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'source': source,
                'page_numbers': page_numbers or [],
                'created_at': now_iso(),
                'created_at_ts': now_ts(),
                'tags': self._detect_tags(chunk.get('text', '')),
                'embedding_model': self._embedding_model_name
            }
        return chunks

    def embed_and_store(self, chunks: List[Dict]) -> int:
        texts = [c['text'] for c in chunks]
        if not texts:
            return 0
        print(f"🧠 Embedding {len(texts)} chunks...")
        embeddings = self.embed_chunks(texts)
        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            metadata = chunk.get('metadata', {})
            if 'chunk_id' not in metadata:
                metadata['chunk_id'] = metadata.get('doc_id', 'doc') + f"::chunk_{i}"
            metadata['text'] = chunk['text'][:4000]
            point = rest.PointStruct(id=metadata['chunk_id'], vector=emb.tolist(), payload=metadata)
            points.append(point)
        print(f"💾 Storing {len(points)} points in Qdrant...")
        self.store_in_qdrant(points)
        return len(points)

# ----------------- evaluation helpers -----------------
def precision_at_k(retrieved: List[str], relevant_set: set, k: int = 5) -> float:
    retrieved_k = retrieved[:k]
    return sum(1 for r in retrieved_k if r in relevant_set) / k

def reciprocal_rank(retrieved: List[str], relevant_set: set) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            return 1.0 / i
    return 0.0

def recall_at_k(retrieved: List[str], relevant_set: set, k: int = 5) -> float:
    retrieved_k = set(retrieved[:k])
    if not relevant_set:
        return 0.0
    return len(retrieved_k.intersection(relevant_set)) / len(relevant_set)
