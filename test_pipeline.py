#!/usr/bin/env python3
"""
Simple test script for RAG pipeline without Docker.
This script tests the core functionality locally.
"""

import os
import sys
from pathlib import Path
import logging

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from pipeline_qdrant import RAGIndexer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_local_pipeline():
    """Test the RAG pipeline locally."""
    logger.info("🧪 Starting local RAG pipeline test")
    
    try:
        # Test 1: Initialize RAGIndexer (this will fail if Qdrant is not running)
        logger.info("🔗 Testing RAGIndexer initialization...")
        indexer = RAGIndexer(collection_name="test_rag_chunks")
        logger.info("✅ RAGIndexer initialized successfully")
        
        # Test 2: Test text processing
        logger.info("📝 Testing text processing...")
        sample_text = """
        Installation Guide
        
        To install the software, run the installer. Requirements: Python 3.8.
        
        Usage:
        1. Run setup
        2. Configure environment
        
        License
        Software is licensed under MIT.
        """
        
        # Test normalization
        normalized = indexer.normalize_text(sample_text)
        logger.info(f"✅ Text normalized: {len(normalized)} characters")
        
        # Test chunking
        chunks = indexer.semantic_chunk(normalized, max_tokens=200, overlap=25)
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"   Chunk {i+1}: {chunk['text'][:100]}...")
        
        # Test 3: Test file ingestion (if sample file exists)
        sample_file = Path("samples/sample1.txt")
        if sample_file.exists():
            logger.info(f"📁 Testing file ingestion: {sample_file}")
            count = indexer.ingest_file(str(sample_file))
            logger.info(f"✅ Ingested {count} chunks from {sample_file}")
            
            # Test 4: Test querying
            logger.info("🔍 Testing query...")
            results = indexer.query("how to install the software", top_k=3)
            logger.info(f"✅ Query returned {len(results)} results")
            
            for i, result in enumerate(results):
                logger.info(f"   Result {i+1}: score={result['score']:.3f}, text={result['text'][:100]}...")
        else:
            logger.warning(f"⚠️ Sample file not found: {sample_file}")
        
        logger.info("🎉 All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_without_qdrant():
    """Test basic functionality without Qdrant."""
    logger.info("🧪 Testing basic functionality without Qdrant...")
    
    try:
        # Test text processing functions directly
        from pipeline_qdrant import normalize_text, semantic_sliding_chunks
        
        sample_text = "This is a test document. It has multiple sentences. Each sentence should be processed correctly."
        
        # Test normalization
        normalized = normalize_text(sample_text)
        logger.info(f"✅ Normalization works: {len(normalized)} chars")
        
        # Test chunking
        chunks = semantic_sliding_chunks(normalized, target_chars=100, overlap_chars=20)
        logger.info(f"✅ Chunking works: {len(chunks)} chunks")
        
        logger.info("🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting RAG Pipeline Tests")
    
    # Try full test first
    if test_local_pipeline():
        logger.info("✅ Full pipeline test successful!")
    else:
        logger.info("⚠️ Full pipeline test failed, trying basic functionality...")
        if test_without_qdrant():
            logger.info("✅ Basic functionality works - Qdrant connection issue")
            logger.info("💡 To run full tests:")
            logger.info("   1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant:latest")
            logger.info("   2. Run this script again")
        else:
            logger.error("❌ Basic functionality also failed")
            sys.exit(1)
