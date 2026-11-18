"""Verificaciones numéricas independientes y del contrato REST."""
import math
from pathlib import Path
import argparse
from datetime import datetime, timezone
import httpx

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from evaluation import load_dataset, metrics_at_k
from preprocessing import preprocess
from retrieval import SearchIndex


@pytest.fixture(scope='module')
def index():
    return SearchIndex.load(Path('artifacts/index'))


def test_manual_models():
    tiny = SearchIndex([{'_id': 'D1', 'title': '', 'text': 'apple apple banana'},
                        {'_id': 'D2', 'title': '', 'text': 'banana carrot'}])
    # apple: df=1; banana: df=2; consulta apple banana.
    ia = 1 + math.log(3 / 2)
    wa = (1 + math.log(2)) * ia
    expected_cos = (wa * ia + 1) / (math.sqrt(wa ** 2 + 1) * math.sqrt(ia ** 2 + 1))
    result = tiny.search('apple banana', 'tfidf', 2)
    assert result[0]['document_id'] == 'D1'
    assert result[0]['score'] == pytest.approx(expected_cos)
    expected_bm = math.log(2) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2.5))
    expected_bm += math.log(1.2) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 3 / 2.5))
    assert tiny.search('apple banana', 'bm25', 2)[0]['score'] == pytest.approx(expected_bm)


def test_manual_metrics():
    judgments = {'A': 2, 'B': 1, 'C': 1}
    result = metrics_at_k(['B', 'X', 'A'], judgments, 3)
    expected = (1 + 3 / math.log2(4)) / (3 + 1 / math.log2(3) + 1 / math.log2(4))
    assert result == pytest.approx({'Precision@3': 2 / 3, 'Recall@3': 2 / 3, 'NDCG@3': expected})
    assert metrics_at_k(['A'], judgments)['Precision@10'] == 0.1
    assert all(v == 0 for v in metrics_at_k([], {}).values())
    assert all(v == 0 for v in metrics_at_k(['X'], {'X': 0}).values())
    with pytest.raises(ValueError):
        metrics_at_k(['A', 'A'], judgments)


def test_preprocessing():
    terms = preprocess("Vitamin B12: 2.5 mg; NOT without β-carotene, risk >= 5%, -3.5. Isn't harmful.")
    for token in ['b12', '2.5', 'not', 'without', 'β', '>=', '5', '%', '-3.5']:
        assert token in terms
    assert preprocess('ＣＡＮＣＥＲ') == preprocess('cancer')


def test_integrity_and_matrices(index):
    docs, queries, qrels = load_dataset()
    assert len(docs) == index.N == 3633
    assert len(queries) == 3237 and len(qrels) == 323
    assert sum(map(len, qrels.values())) == 12334
    assert index.doc_ids == [d['_id'] for d in docs]
    assert np.all(index.doc_lengths == index.tf.sum(axis=1).A1)
    assert np.all(index.df == (index.tf > 0).sum(axis=0).A1)


@pytest.mark.parametrize('model', ['bm25', 'tfidf'])
def test_against_independent_full_corpus_formula(index, model):
    query = 'vitamin vitamin B12 cancer risk'
    scores, candidates = index.score(query, model)
    # Comparación independiente: coseno mediante producto matricial completo;
    # BM25 escalar sin leer postings para una muestra determinista de documentos.
    if model == 'tfidf':
        q = index.query_vector(query)
        expected = (index.tfidf @ q.T).toarray().ravel()
        denominator = index.doc_norms * math.sqrt(q.multiply(q).sum())
        expected = np.divide(expected, denominator, out=np.zeros_like(expected), where=denominator > 0)
        np.testing.assert_allclose(scores, expected, atol=1e-12)
    else:
        terms = set(preprocess(query)) & index.vocabulary.keys()
        for p in np.random.default_rng(753).choice(index.N, 100, replace=False):
            value = 0.0
            for t in terms:
                col = index.vocabulary[t]
                f = float(index.tf[p, col])
                df = int((index.tf[:, col] > 0).sum())
                value += math.log(1 + (index.N - df + .5) / (df + .5)) * f * 2.2 / (f + 1.2 * (.25 + .75 * index.doc_lengths[p] / index.avgdl))
            assert scores[p] == pytest.approx(value, abs=1e-12)
    assert set(candidates) == set(np.flatnonzero(scores))


def test_save_load_and_ties(tmp_path):
    original = SearchIndex([{'_id': did, 'title': 'apple', 'text': ''} for did in ['Z', 'A']])
    original.save(tmp_path)
    loaded = SearchIndex.load(tmp_path)
    for model in ['tfidf', 'bm25']:
        assert loaded.search('apple', model) == original.search('apple', model)
        assert [r['document_id'] for r in loaded.search('apple', model)] == ['A', 'Z']
        assert loaded.search('zzyyxx', model) == []


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as connected:
        yield connected


def test_health(client):
    result = client.get('/health')
    assert result.status_code == 200
    assert result.json()['documents'] == 3633


@pytest.mark.parametrize('model', ['tfidf', 'bm25'])
@pytest.mark.parametrize('k', [1, 7, 10, 100, 99999])
def test_api_contract(client, index, model, k):
    for query in ['Breast Cancer Cells Feed on Cholesterol', 'fermented chickpeas exercise inflammation 2026']:
        response = client.post('/search', json={'query': query, 'model': model, 'top_k': k})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {'model', 'results'}
        results = body['results']
        assert results == index.search(query, model, k)
        assert len(results) <= min(k, index.N)
        assert len({r['document_id'] for r in results}) == len(results)
        assert [r['rank'] for r in results] == list(range(1, len(results) + 1))
        assert all(math.isfinite(r['score']) and r['document_id'] in index.id_to_position for r in results)
        assert [r['score'] for r in results] == sorted([r['score'] for r in results], reverse=True)
        assert all(set(r) == {'rank', 'document_id', 'score', 'title', 'snippet'} for r in results)


@pytest.mark.parametrize('override', [
    {'query': ''}, {'query': '  '}, {'query': None}, {'query': 5}, {'model': 'unknown'},
    {'top_k': 0}, {'top_k': -1}, {'top_k': 1.5}, {'top_k': True}, {'top_k': '10'}, {'extra': 1}
])
def test_bad_requests(client, override):
    payload = {'query': 'cancer', 'model': 'bm25', 'top_k': 10, **override}
    assert client.post('/search', json=payload).status_code == 422


def test_unknown_and_missing(client):
    assert client.post('/search', json={'query': 'zxqvprsnm', 'model': 'bm25', 'top_k': 10}).json()['results'] == []
    assert client.post('/search', json={'query': 'cancer'}).status_code == 422
    assert client.post('/search', content='invalid', headers={'Content-Type': 'application/json'}).status_code == 422



def verify(base_url: str) -> dict:
    index = SearchIndex.load(Path(__file__).resolve().parents[1] / 'artifacts/index')
    _, queries, qrels = load_dataset()
    official = queries[sorted(qrels)[0]]
    records = []
    with httpx.Client(base_url=base_url.rstrip('/'), timeout=180, follow_redirects=True, trust_env=False) as client:
        health = client.get('/health')
        assert health.status_code == 200 and health.json()['documents'] == index.N
        records.append({'method': 'GET', 'path': '/health', 'status': health.status_code, 'body': health.json()})
        for model in ['tfidf', 'bm25']:
            for query in [official, 'dietary lentils magnesium sleep performance', 'zxqvprsnm']:
                for k in [1, 10, 23]:
                    payload = {'query': query, 'model': model, 'top_k': k}
                    result = client.post('/search', json=payload)
                    assert result.status_code == 200
                    assert result.json() == {'model': model, 'results': index.search(query, model, k)}
                    records.append({'method': 'POST', 'path': '/search', 'request': payload,
                                    'status': result.status_code, 'body': result.json()})
        for patch in [{'model': 'invalid'}, {'query': ''}, {'top_k': 0}, {'top_k': True}]:
            payload = {'query': 'cancer', 'model': 'bm25', 'top_k': 10, **patch}
            result = client.post('/search', json=payload)
            assert result.status_code == 422
            records.append({'method': 'POST', 'path': '/search', 'request': payload, 'status': result.status_code})
    return {'base_url': base_url, 'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': len(records), 'records': records}


def verify_local():
    """Inicia un servidor temporal, hace pruebas HTTP reales y lo cierra."""
    import socket
    import subprocess
    import sys
    import time

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app:app', '--host',
        '127.0.0.1', '--port', str(port)], cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        for attempt in range(120):
            if server.poll() is not None:
                raise RuntimeError('El servidor de prueba no pudo iniciar')
            try:
                if httpx.get(url + '/health', timeout=1).status_code == 200:
                    return verify(url)
            except httpx.HTTPError:
                pass
            time.sleep(.25)
        raise TimeoutError('El servidor de prueba no respondió')
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verificar una API real contra el índice local')
    parser.add_argument('url', nargs='?', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    report = verify(args.url)
    print(f"{report['checks']} comprobaciones HTTP correctas: {args.url}")
