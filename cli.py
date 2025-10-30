import argparse
from pipeline_qdrant import RAGIndexer
from pathlib import Path

def ingest_path(indexer: RAGIndexer, path: str):
    p = Path(path)
    files = []
    if p.is_dir():
        for ext in ('*.pdf','*.docx','*.html','*.htm','*.txt'):
            files.extend(list(p.glob(ext)))
    elif p.is_file():
        files = [p]
    else:
        print('Path not found:', path)
        return
    total = 0
    for f in files:
        print('Ingesting', f)
        try:
            cnt = indexer.ingest_file(str(f))
            print('  chunks:', cnt)
            total += cnt
        except Exception as e:
            print('  ERROR:', e)
    print('Ingested total chunks:', total)

def run_query(indexer: RAGIndexer, query: str, top_k: int):
    res = indexer.query(query, top_k=top_k, filter_payload=None)
    for i, r in enumerate(res, start=1):
        payload = r['payload']
        print(f"\n[{i}] score={r['score']:.4f} doc={payload.get('doc_id')} src={payload.get('source')}")
        if payload.get('heading'):
            print('  heading:', payload.get('heading'))
        print('  text:', payload.get('text', '')[:600].replace('\n',' '), '...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='rag-cli')
    sub = parser.add_subparsers(dest='cmd')

    p_ingest = sub.add_parser('ingest')
    p_ingest.add_argument('path')
    p_ingest.add_argument('--collection', default='rag_chunks')

    p_query = sub.add_parser('query')
    p_query.add_argument('query')
    p_query.add_argument('--top_k', type=int, default=5)
    p_query.add_argument('--collection', default='rag_chunks')

    args = parser.parse_args()
    indexer = RAGIndexer(collection_name=(args.collection if hasattr(args, 'collection') else 'rag_chunks'))

    if args.cmd == 'ingest':
        ingest_path(indexer, args.path)
    elif args.cmd == 'query':
        run_query(indexer, args.query, args.top_k)
    else:
        parser.print_help()
