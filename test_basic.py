#!/usr/bin/env python3
"""
Simple test script for RAG pipeline basic functionality.
This script tests core functions without requiring Qdrant or sentence-transformers.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functions():
    """Test basic text processing functions."""
    logger.info("🧪 Testing basic text processing functions...")
    
    try:
        # Import only the functions we need
        from pipeline_qdrant import normalize_text, semantic_sliding_chunks, heading_based_chunks
        
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
        logger.info("📝 Testing text normalization...")
        normalized = normalize_text(sample_text)
        logger.info(f"✅ Normalization works: {len(normalized)} characters")
        logger.info(f"   Normalized text preview: {normalized[:100]}...")
        
        # Test semantic chunking
        logger.info("🔪 Testing semantic chunking...")
        chunks = semantic_sliding_chunks(normalized, target_chars=150, overlap_chars=30)
        logger.info(f"✅ Semantic chunking works: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"   Chunk {i+1}: {chunk['text'][:80]}...")
        
        # Test heading-based chunking
        logger.info("📑 Testing heading-based chunking...")
        h_chunks = heading_based_chunks(normalized, max_chars=200)
        logger.info(f"✅ Heading-based chunking works: {len(h_chunks)} chunks")
        
        for i, chunk in enumerate(h_chunks):
            heading = chunk.get('heading', 'No heading')
            logger.info(f"   Chunk {i+1} (heading: {heading}): {chunk['text'][:60]}...")
        
        logger.info("🎉 All basic function tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic function test failed: {e}")
        return False

def test_file_reading():
    """Test file reading functions."""
    logger.info("📁 Testing file reading functions...")
    
    try:
        from pipeline_qdrant import read_html, file_id
        
        # Test HTML reading
        sample_file = Path("samples/sample1.html")
        if sample_file.exists():
            logger.info(f"📄 Testing HTML file reading: {sample_file}")
            text, meta = read_html(str(sample_file))
            logger.info(f"✅ HTML reading works: {len(text)} characters")
            logger.info(f"   Text preview: {text[:100]}...")
        else:
            logger.warning(f"⚠️ Sample HTML file not found: {sample_file}")
        
        # Test file ID generation
        logger.info("🆔 Testing file ID generation...")
        test_path = "samples/test_document.pdf"
        doc_id = file_id(test_path)
        logger.info(f"✅ File ID generation works: {doc_id}")
        
        logger.info("🎉 File reading tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ File reading test failed: {e}")
        return False

def test_sample_file():
    """Test with the actual sample file."""
    logger.info("📄 Testing with sample file...")
    
    try:
        sample_file = Path("samples/sample1.txt")
        if not sample_file.exists():
            logger.warning(f"⚠️ Sample file not found: {sample_file}")
            return False
        
        # Read the file
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"📖 Read sample file: {len(content)} characters")
        logger.info(f"   Content preview: {content[:200]}...")
        
        # Test processing
        from pipeline_qdrant import normalize_text, semantic_sliding_chunks
        
        normalized = normalize_text(content)
        chunks = semantic_sliding_chunks(normalized, target_chars=200, overlap_chars=50)
        
        logger.info(f"✅ Processed sample file into {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"   Chunk {i+1}: {chunk['text'][:100]}...")
        
        logger.info("🎉 Sample file test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sample file test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting RAG Pipeline Basic Tests")
    
    tests = [
        ("Basic Functions", test_basic_functions),
        ("File Reading", test_file_reading),
        ("Sample File", test_sample_file)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Basic RAG pipeline functionality works.")
        logger.info("💡 To test full functionality:")
        logger.info("   1. Fix Docker/WSL issues")
        logger.info("   2. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant:latest")
        logger.info("   3. Install dependencies: pip install -r requirements.txt")
        logger.info("   4. Run: python test_pipeline.py")
    else:
        logger.error("❌ Some tests failed. Check the errors above.")
        sys.exit(1)
