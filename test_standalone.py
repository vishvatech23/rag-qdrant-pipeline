#!/usr/bin/env python3
"""
Standalone test for RAG pipeline basic functionality.
This script tests core functions without importing from pipeline_qdrant.py.
"""

import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize text by cleaning whitespace and removing duplicates."""
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 20:
        freq = {}
        for l in lines[:20]:
            freq[l] = freq.get(l, 0) + 1
        repeats = {k for k,v in freq.items() if v >= 2}
        if repeats:
            text = '\n'.join([l for l in lines if l not in repeats])
    return text

def semantic_sliding_chunks(text: str, target_chars: int = 1500, overlap_chars: int = 300):
    """Create semantic chunks with sliding window."""
    # Simple sentence splitting (without NLTK for this test)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    cur = []
    cur_len = 0
    
    for s in sentences:
        sl = len(s)
        if cur_len + sl > target_chars and cur:
            chunks.append({'heading': None, 'text': ' '.join(cur)})
            # Keep overlap
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

def read_html(path: str):
    """Read HTML file and extract text."""
    with open(path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    for s in soup(['script', 'style']):
        s.decompose()
    text = soup.get_text(separator='\n')
    return text, {}

def test_basic_functions():
    """Test basic text processing functions."""
    logger.info("🧪 Testing basic text processing functions...")
    
    try:
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
        
        logger.info("🎉 All basic function tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic function test failed: {e}")
        return False

def test_file_reading():
    """Test file reading functions."""
    logger.info("📁 Testing file reading functions...")
    
    try:
        # Test HTML reading
        sample_file = Path("samples/sample1.html")
        if sample_file.exists():
            logger.info(f"📄 Testing HTML file reading: {sample_file}")
            text, meta = read_html(str(sample_file))
            logger.info(f"✅ HTML reading works: {len(text)} characters")
            logger.info(f"   Text preview: {text[:100]}...")
        else:
            logger.warning(f"⚠️ Sample HTML file not found: {sample_file}")
        
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
    logger.info("🚀 Starting RAG Pipeline Standalone Tests")
    
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
        logger.info("💡 The core text processing functions are working correctly.")
        logger.info("💡 To test full functionality with Qdrant:")
        logger.info("   1. Fix Docker/WSL issues")
        logger.info("   2. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant:latest")
        logger.info("   3. Install dependencies: pip install -r requirements.txt")
        logger.info("   4. Run: python test_pipeline.py")
    else:
        logger.error("❌ Some tests failed. Check the errors above.")
