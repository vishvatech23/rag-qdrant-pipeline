from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path
from typing import Optional, List
import logging
from datetime import datetime
from pipeline_qdrant import RAGIndexer

# ------------------------------------------------------------
# Configure logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# FastAPI initialization
# ------------------------------------------------------------
app = FastAPI(
    title="RAG Qdrant Pipeline",
    description="Retrieval-Augmented Generation pipeline with Qdrant vector database",
    version="1.0.0"
)

# ------------------------------------------------------------
# Initialize RAG Indexer
# ------------------------------------------------------------
try:
    indexer = RAGIndexer(collection_name="rag_chunks")
    logger.info("✅ RAG Indexer initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG Indexer: {e}")
    indexer = None

# ------------------------------------------------------------
# File upload directory setup
# ------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Supported file types
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.html', '.htm', '.txt'}

# ------------------------------------------------------------
# API ROUTES
# ------------------------------------------------------------

@app.get("/")
async def root():
    """Basic Health Check."""
    return {
        "status": "healthy",
        "service": "RAG Qdrant Pipeline",
        "version": "1.0.0",
        "qdrant_connected": indexer is not None
    }


@app.get("/health")
async def health_check():
    """Detailed Health Check for Qdrant connection."""
    if indexer is None:
        raise HTTPException(status_code=503, detail="RAG Indexer not initialized")

    try:
        collections = indexer.qclient.get_collections()
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collections": [c.name for c in collections.collections],
            "active_collection": indexer.collection_name
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant connection failed: {str(e)}")


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a document (PDF, DOCX, HTML, or TXT) into the RAG pipeline.
    """
    if indexer is None:
        raise HTTPException(status_code=503, detail="RAG Indexer not initialized")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        filepath = UPLOAD_DIR / file.filename
        logger.info(f"📁 Saving uploaded file: {file.filename}")

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process file
        logger.info(f"🔄 Processing file: {filepath}")
        chunks_ingested = indexer.ingest_file(str(filepath))

        # Generate doc_id from filename
        doc_id = indexer._make_meta(file.filename, str(filepath), "", None)['doc_id']

        # Clean up uploaded file after processing
        filepath.unlink(missing_ok=True)

        logger.info(f"✅ Successfully ingested {chunks_ingested} chunks from {file.filename}")

        return {
            "status": "success",
            "file": file.filename,
            "chunks_ingested": chunks_ingested,
            "doc_id": doc_id
        }

    except ValueError as e:
        logger.error(f"❌ File processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("❌ Unexpected error during ingestion")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/query")
async def query_docs(
    q: str = Query(..., description="Query text to search for"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
    doc_id: Optional[str] = Query(None, description="Filter by document ID"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    date_from: Optional[str] = Query(None, description="Filter results from this date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter results until this date (ISO format)")
):
    """
    Query the RAG pipeline for relevant document chunks.
    """
    if indexer is None:
        raise HTTPException(status_code=503, detail="RAG Indexer not initialized")

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")

    # Parse tags
    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    # Convert date filters to epoch timestamps
    date_from_ts, date_to_ts = None, None
    try:
        if date_from:
            date_from_ts = int(datetime.fromisoformat(date_from).timestamp())
        if date_to:
            date_to_ts = int(datetime.fromisoformat(date_to).timestamp())
    except Exception as e:
        logger.error("❌ Date parsing error", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). Error: {e}")

    try:
        logger.info(f"🔍 Querying: '{q[:60]}' (top_k={top_k}) filters={{'doc_id': {doc_id}, 'tags': {tag_list}}}")

        results = indexer.query(
            query_text=q,
            top_k=top_k,
            doc_id=doc_id,
            tags=tag_list,
            date_from_ts=date_from_ts,
            date_to_ts=date_to_ts
        )

        logger.info(f"✅ Query returned {len(results)} results")

        return {
            "status": "success",
            "query": q,
            "results": results,
            "total_results": len(results),
            "filters": {
                "doc_id": doc_id,
                "tags": tag_list,
                "date_from": date_from,
                "date_to": date_to
            }
        }

    except Exception as e:
        logger.exception("❌ Query error")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/collections")
async def list_collections():
    """List all Qdrant collections."""
    if indexer is None:
        raise HTTPException(status_code=503, detail="RAG Indexer not initialized")

    try:
        collections = indexer.qclient.get_collections()
        return {
            "collections": [
                {
                    "name": c.name,
                    "points_count": getattr(c, "points_count", None),
                    "status": getattr(c, "status", None)
                }
                for c in collections.collections
            ]
        }
    except Exception as e:
        logger.exception("❌ Failed to list collections")
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a Qdrant collection."""
    if indexer is None:
        raise HTTPException(status_code=503, detail="RAG Indexer not initialized")

    try:
        indexer.qclient.delete_collection(collection_name)
        logger.info(f"🗑️ Deleted collection: {collection_name}")
        return {"status": "success", "message": f"Collection '{collection_name}' deleted"}
    except Exception as e:
        logger.exception("❌ Failed to delete collection")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
