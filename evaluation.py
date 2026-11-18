"""Consultas/qrels test y métricas; no se importa desde la API."""
import csv
import json
import math

import pandas as pd

from retrieval import DATA_DIR, load_documents


def load_dataset():
    documents = load_documents()
    with (DATA_DIR / 'queries.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    queries = {q['_id']: q['text'] for q in rows}
    if len(queries) != len(rows) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in queries.items()):
        raise ValueError('Consultas inválidas o duplicadas')
    doc_ids = {d['_id'] for d in documents}
    qrels = {}
    with (DATA_DIR / 'qrels/test.tsv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            qid, did, grade = row['query-id'], row['corpus-id'], int(row['score'])
            judgments = qrels.setdefault(qid, {})
            if qid not in queries or did not in doc_ids or grade < 0 or did in judgments:
                raise ValueError(f'Juicio inválido: {row}')
            judgments[did] = grade
    if not qrels:
        raise ValueError('No existen juicios test')
    return documents, queries, qrels


def metrics_at_k(ranking, judgments, k=10):
    if k < 1 or len(set(ranking)) != len(ranking):
        raise ValueError('k inválido o ranking con duplicados')
    grades = [judgments.get(did, 0) for did in ranking[:k]]
    relevant = sum(g > 0 for g in judgments.values())
    hits = sum(g > 0 for g in grades)
    dcg = sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, 1))
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(ideal, 1))
    return {f'Precision@{k}': hits / k, f'Recall@{k}': hits / relevant if relevant else 0.0,
            f'NDCG@{k}': dcg / idcg if idcg else 0.0}


def evaluate(index, queries, qrels, k=10):
    rows, runs = [], {'tfidf': {}, 'bm25': {}}
    for model in runs:
        for qid in sorted(qrels):
            result = index.search(queries[qid], model, k)
            runs[model][qid] = result
            rows.append({'model': model, 'query_id': qid,
                         **metrics_at_k([r['document_id'] for r in result], qrels[qid], k)})
    per_query = pd.DataFrame(rows)
    summary = per_query.groupby('model', sort=False)[[f'Precision@{k}', f'Recall@{k}', f'NDCG@{k}']].mean()
    return per_query, summary, runs
