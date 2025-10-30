#!/usr/bin/env python3
"""
RAG Pipeline Demo Script
--------------------------------
Runs a full test sequence for your FastAPI + Qdrant setup.
"""

import requests
import json
import time
from pathlib import Path

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Dummy:
        def __getattr__(self, name): return ''
    Fore = Style = Dummy()


def test_api_endpoints():
    """Run all endpoint tests."""
    base_url = "http://localhost:8000"

    print(Fore.CYAN + "\n🚀 RAG Pipeline Demo - Full System Check")
    print("=" * 60)

    # 1️⃣ Health check
    print(Fore.YELLOW + "\n[1] Testing basic health check...")
    try:
        res = requests.get(f"{base_url}/")
        if res.status_code == 200:
            print(Fore.GREEN + "✅ API is healthy")
            print(json.dumps(res.json(), indent=4))
        else:
            print(Fore.RED + f"❌ Health check failed ({res.status_code})")
            return False
    except Exception as e:
        print(Fore.RED + f"❌ Could not connect to API: {e}")
        print("💡 Run: uvicorn main:app --host 0.0.0.0 --port 8000")
        return False

    # 2️⃣ Detailed health check
    print(Fore.YELLOW + "\n[2] Testing detailed health check...")
    try:
        res = requests.get(f"{base_url}/health")
        if res.status_code == 200:
            data = res.json()
            print(Fore.GREEN + "✅ Qdrant connected")
            print(f"   Active collection: {data.get('active_collection')}")
            print(f"   Collections: {data.get('collections', [])}")
        else:
            print(Fore.RED + f"❌ Detailed health check failed ({res.status_code})")
    except Exception as e:
        print(Fore.RED + f"❌ Error: {e}")

    # 3️⃣ List collections
    print(Fore.YELLOW + "\n[3] Testing collections list...")
    try:
        res = requests.get(f"{base_url}/collections")
        if res.status_code == 200:
            data = res.json()
            print(Fore.GREEN + "✅ Collections retrieved:")
            for col in data.get('collections', []):
                print(f"   - {col['name']}: {col.get('points_count', '?')} points")
        else:
            print(Fore.RED + f"❌ Failed to list collections ({res.status_code})")
            print(Fore.LIGHTBLACK_EX + res.text)
    except Exception as e:
        print(Fore.RED + f"❌ Error fetching collections: {e}")

    # 4️⃣ Ingest a document
    print(Fore.YELLOW + "\n[4] Testing document ingestion...")
    sample_file = Path("samples/sample1.txt")
    if not sample_file.exists():
        print(Fore.RED + f"⚠️ Sample file not found: {sample_file}")
        return False

    try:
        with open(sample_file, "rb") as f:
            res = requests.post(f"{base_url}/ingest", files={"file": f})

        if res.status_code == 200:
            data = res.json()
            print(Fore.GREEN + "✅ Document ingested successfully")
            print(f"   File: {data.get('file')}")
            print(f"   Chunks: {data.get('chunks_ingested')}")
            print(f"   Doc ID: {data.get('doc_id')}")
        else:
            print(Fore.RED + f"❌ Ingestion failed ({res.status_code})")
            print(Fore.LIGHTBLACK_EX + res.text)
            return False
    except Exception as e:
        print(Fore.RED + f"❌ Ingestion error: {e}")
        return False

    # 5️⃣ Query functionality
    print(Fore.YELLOW + "\n[5] Testing query functionality...")
    test_queries = [
        "how to install the software",
        "python requirements",
        "license information",
        "usage instructions",
    ]

    for query in test_queries:
        print(Fore.CYAN + f"\n🔍 Query: '{query}'")
        try:
            params = {"q": query, "top_k": 3}
            res = requests.get(f"{base_url}/query", params=params)

            if res.status_code == 200:
                data = res.json()
                print(Fore.GREEN + f"✅ Query successful ({data.get('total_results', 0)} results)")
                for i, r in enumerate(data.get("results", [])[:3], start=1):
                    snippet = r["text"].replace("\n", " ")[:120]
                    print(Fore.WHITE + f"   [{i}] Score: {r['score']:.3f}")
                    print(Fore.LIGHTBLACK_EX + f"       {snippet}...")
            else:
                print(Fore.RED + f"❌ Query failed ({res.status_code})")
                print(Fore.LIGHTBLACK_EX + res.text)
        except Exception as e:
            print(Fore.RED + f"❌ Query error: {e}")

    print(Fore.GREEN + "\n🎉 Demo completed successfully!\n")
    return True


def show_usage_examples():
    """Display example API calls."""
    print(Fore.CYAN + "\n📘 Example API Commands")
    print("=" * 50)

    examples = [
        {
            "title": "Ingest a document",
            "cmd": 'curl -F "file=@samples/sample1.txt" http://localhost:8000/ingest',
        },
        {
            "title": "Query the system",
            "cmd": 'curl "http://localhost:8000/query?q=how%20to%20install&top_k=3"',
        },
        {
            "title": "Filter by tags",
            "cmd": 'curl "http://localhost:8000/query?q=python&tags=code,installation"',
        },
        {
            "title": "List collections",
            "cmd": "curl http://localhost:8000/collections",
        },
        {
            "title": "Health check",
            "cmd": "curl http://localhost:8000/health",
        },
    ]

    for e in examples:
        print(Fore.YELLOW + f"\n🔹 {e['title']}")
        print(Fore.WHITE + f"   {e['cmd']}")


def main():
    print(Fore.CYAN + "\n🎯 RAG Pipeline Automated Test Suite")
    print("=" * 60)

    success = test_api_endpoints()

    if success:
        show_usage_examples()
        print(Fore.GREEN + "\n💡 Next Steps:")
        print(Fore.WHITE + "   1. Upload your own documents to test further.")
        print(Fore.WHITE + "   2. Try queries using the FastAPI docs → http://localhost:8000/docs")
        print(Fore.WHITE + "   3. If you face issues, check backend logs for stack traces.")
    else:
        print(Fore.RED + "\n❌ Demo failed.")
        print(Fore.WHITE + "   Ensure Qdrant and API are running properly:")
        print(Fore.LIGHTBLACK_EX + "   docker run -p 6333:6333 qdrant/qdrant:latest")
        print(Fore.LIGHTBLACK_EX + "   uvicorn main:app --host 0.0.0.0 --port 8000")
        print(Fore.LIGHTBLACK_EX + "   pip install -r requirements.txt")


if __name__ == "__main__":
    main()
