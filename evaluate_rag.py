#!/usr/bin/env python3
"""
RAG Pipeline Evaluation Script

This script evaluates the retrieval performance of the RAG pipeline using
standard information retrieval metrics: Precision@K, Recall@K, and MRR.
"""

import json
import time
from typing import List, Dict, Set, Tuple
from pathlib import Path
import argparse
import logging
from pipeline_qdrant import RAGIndexer, precision_at_k, recall_at_k, reciprocal_rank

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluator for RAG pipeline performance."""
    
    def __init__(self, indexer: RAGIndexer):
        self.indexer = indexer
        
    def create_test_dataset(self) -> List[Dict]:
        """Create a test dataset with queries and expected relevant chunks."""
        return [
            {
                "query": "how to install the software",
                "expected_chunks": ["install", "setup", "requirements"],
                "description": "Installation instructions"
            },
            {
                "query": "software license",
                "expected_chunks": ["license", "MIT", "permission"],
                "description": "License information"
            },
            {
                "query": "usage instructions",
                "expected_chunks": ["usage", "run", "configure"],
                "description": "Usage documentation"
            },
            {
                "query": "python requirements",
                "expected_chunks": ["python", "3.8", "requirements"],
                "description": "Python version requirements"
            }
        ]
    
    def evaluate_query(self, query: str, expected_keywords: List[str], top_k: int = 5) -> Dict:
        """Evaluate a single query against expected keywords."""
        logger.info(f"🔍 Evaluating query: '{query}'")
        
        # Get results from RAG pipeline
        start_time = time.time()
        results = self.indexer.query(query, top_k=top_k)
        query_time = time.time() - start_time
        
        # Extract retrieved text
        retrieved_texts = [r['text'] for r in results]
        
        # Check relevance based on keyword presence
        relevant_chunks = set()
        for i, text in enumerate(retrieved_texts):
            text_lower = text.lower()
            for keyword in expected_keywords:
                if keyword.lower() in text_lower:
                    relevant_chunks.add(f"chunk_{i}")
                    break
        
        # Calculate metrics
        precision = precision_at_k(list(relevant_chunks), relevant_chunks, top_k)
        recall = recall_at_k(list(relevant_chunks), relevant_chunks, top_k)
        mrr = reciprocal_rank(list(relevant_chunks), relevant_chunks)
        
        return {
            "query": query,
            "query_time": query_time,
            "total_results": len(results),
            "relevant_results": len(relevant_chunks),
            "precision_at_k": precision,
            "recall_at_k": recall,
            "mrr": mrr,
            "results": [
                {
                    "rank": i + 1,
                    "score": r['score'],
                    "text": r['text'][:200] + "..." if len(r['text']) > 200 else r['text'],
                    "is_relevant": f"chunk_{i}" in relevant_chunks
                }
                for i, r in enumerate(results)
            ]
        }
    
    def run_evaluation(self, test_dataset: List[Dict], top_k: int = 5) -> Dict:
        """Run full evaluation on test dataset."""
        logger.info(f"🚀 Starting RAG evaluation with {len(test_dataset)} queries")
        
        results = []
        total_precision = 0
        total_recall = 0
        total_mrr = 0
        total_query_time = 0
        
        for test_case in test_dataset:
            eval_result = self.evaluate_query(
                test_case["query"], 
                test_case["expected_chunks"], 
                top_k
            )
            results.append(eval_result)
            
            total_precision += eval_result["precision_at_k"]
            total_recall += eval_result["recall_at_k"]
            total_mrr += eval_result["mrr"]
            total_query_time += eval_result["query_time"]
            
            logger.info(f"✅ Query '{test_case['query']}' - P@{top_k}: {eval_result['precision_at_k']:.3f}, "
                      f"R@{top_k}: {eval_result['recall_at_k']:.3f}, MRR: {eval_result['mrr']:.3f}")
        
        # Calculate averages
        avg_precision = total_precision / len(test_dataset)
        avg_recall = total_recall / len(test_dataset)
        avg_mrr = total_mrr / len(test_dataset)
        avg_query_time = total_query_time / len(test_dataset)
        
        summary = {
            "evaluation_summary": {
                "total_queries": len(test_dataset),
                "top_k": top_k,
                "average_precision_at_k": avg_precision,
                "average_recall_at_k": avg_recall,
                "average_mrr": avg_mrr,
                "average_query_time": avg_query_time,
                "total_evaluation_time": total_query_time
            },
            "detailed_results": results
        }
        
        logger.info(f"📊 Evaluation Summary:")
        logger.info(f"   Average Precision@{top_k}: {avg_precision:.3f}")
        logger.info(f"   Average Recall@{top_k}: {avg_recall:.3f}")
        logger.info(f"   Average MRR: {avg_mrr:.3f}")
        logger.info(f"   Average Query Time: {avg_query_time:.3f}s")
        
        return summary
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {output_file}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline performance")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--qdrant-host", type=str, default="http://localhost:6333", help="Qdrant host URL")
    parser.add_argument("--collection", type=str, default="rag_chunks", help="Qdrant collection name")
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG indexer
        logger.info(f"🔗 Connecting to Qdrant at {args.qdrant_host}")
        indexer = RAGIndexer(
            qdrant_host=args.qdrant_host,
            collection_name=args.collection
        )
        
        # Create evaluator
        evaluator = RAGEvaluator(indexer)
        
        # Create test dataset
        test_dataset = evaluator.create_test_dataset()
        
        # Run evaluation
        results = evaluator.run_evaluation(test_dataset, top_k=args.top_k)
        
        # Save results
        evaluator.save_results(results, args.output)
        
        logger.info("✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
