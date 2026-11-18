"""Carga de NFCorpus, índice invertido y los dos modelos de recuperación."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from preprocessing import preprocess

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'nfcorpus/nfcorpus'
INDEX_DIR = ROOT / 'artifacts/index'


def load_documents(directory=DATA_DIR):
    with (Path(directory) / 'corpus.jsonl').open(encoding='utf-8') as stream:
        documents = [json.loads(line) for line in stream]
    ids = [doc['_id'] for doc in documents]
    if not documents or len(set(ids)) != len(ids):
        raise ValueError('El corpus está vacío o contiene IDs duplicados')
    if any(not isinstance(doc.get(field), str) or (field == '_id' and not doc[field])
           for doc in documents for field in ['_id', 'title', 'text']):
        raise ValueError('Los documentos deben conservar ID, título y texto como cadenas')
    return documents


class SearchIndex:
    def __init__(self, documents, k1=1.2, b=0.75, *, terms=None, tf=None):
        if not documents or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Corpus o parámetros BM25 inválidos')
        self.documents = documents
        self.doc_ids = [doc['_id'] for doc in documents]
        self.id_to_position = {did: p for p, did in enumerate(self.doc_ids)}
        if len(self.id_to_position) != len(documents):
            raise ValueError('IDs duplicados')
        self.N, self.k1, self.b = len(documents), k1, b

        if tf is None:
            # TF de título + texto, manteniendo el documento completo.
            counts = [Counter(preprocess(d['title'] + '\n' + d['text'])) for d in documents]
            terms = sorted({term for doc in counts for term in doc})
            vocabulary = {term: col for col, term in enumerate(terms)}
            rows, cols, values = [], [], []
            for position, doc in enumerate(counts):
                for term, frequency in doc.items():
                    rows.append(position)
                    cols.append(vocabulary[term])
                    values.append(frequency)
            tf = sparse.csr_matrix((values, (rows, cols)), shape=(self.N, len(terms)), dtype=float)

        # La misma preparación sirve al construir y al cargar el índice.
        self.vocabulary = {term: col for col, term in enumerate(terms)}
        self.tf = tf.tocsr()
        self.doc_lengths = self.tf.sum(axis=1).A1.astype(int)
        self.avgdl = float(self.doc_lengths.mean())
        csc = self.tf.tocsc()
        self.postings = {}
        for term, col in self.vocabulary.items():
            start, end = csc.indptr[col:col + 2]
            self.postings[term] = np.column_stack((csc.indices[start:end], csc.data[start:end])).astype(int)
        self.df = np.diff(csc.indptr)
        self.idf = 1 + np.log((self.N + 1) / (self.df + 1))
        self.bm25_idf = np.log1p((self.N - self.df + 0.5) / (self.df + 0.5))
        weights = self.tf.copy()
        weights.data = 1 + np.log(weights.data)
        self.tfidf = weights.multiply(self.idf).tocsr()
        self.tfidf_csc = self.tfidf.tocsc()
        self.doc_norms = np.sqrt(self.tfidf.multiply(self.tfidf).sum(axis=1)).A1

    def query_vector(self, query):
        counts = Counter(preprocess(query))
        terms = [term for term in counts if term in self.vocabulary]
        columns = [self.vocabulary[term] for term in terms]
        values = [(1 + np.log(counts[term])) * self.idf[col] for term, col in zip(terms, columns)]
        return sparse.csr_matrix((values, ([0] * len(columns), columns)), shape=(1, len(self.vocabulary)))

    def score(self, query, model):
        if model not in {'tfidf', 'bm25'}:
            raise ValueError('model debe ser tfidf o bm25')
        terms = sorted(set(preprocess(query)) & self.vocabulary.keys())
        scores = np.zeros(self.N)
        if not terms:
            return scores, np.array([], dtype=int)
        candidates = np.unique(np.concatenate([self.postings[t][:, 0] for t in terms]))

        if model == 'tfidf':
            vector = self.query_vector(query)
            norm = np.sqrt(vector.multiply(vector).sum())
            for col, query_weight in zip(vector.indices, vector.data):
                start, end = self.tfidf_csc.indptr[col:col + 2]
                positions = self.tfidf_csc.indices[start:end]
                scores[positions] += self.tfidf_csc.data[start:end] * query_weight
            denominator = self.doc_norms[candidates] * norm
            scores[candidates] = np.divide(scores[candidates], denominator,
                out=np.zeros_like(denominator), where=denominator > 0)
        else:
            for term in terms:
                positions, frequencies = self.postings[term].T
                relative_length = self.doc_lengths[positions] / self.avgdl
                denominator = frequencies + self.k1 * (1 - self.b + self.b * relative_length)
                scores[positions] += self.bm25_idf[self.vocabulary[term]] * frequencies * (self.k1 + 1) / denominator
        return scores, candidates

    def search(self, query, model='bm25', top_k=10):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query no puede estar vacía')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k debe ser un entero positivo')
        scores, candidates = self.score(query, model)
        # Desempate reproducible por ID original.
        ordered = sorted(candidates.tolist(), key=lambda p: (-scores[p], self.doc_ids[p]))[:top_k]
        return [{'rank': rank, 'document_id': self.doc_ids[p], 'score': float(scores[p]),
                 'title': self.documents[p]['title'], 'snippet': self.documents[p]['text'][:300]}
                for rank, p in enumerate(ordered, 1)]

    def explain(self, query, document_id):
        """Contribuciones por término para interpretar los ejemplos del notebook."""
        p = self.id_to_position[document_id]
        vector = self.query_vector(query)
        qnorm = np.sqrt(vector.multiply(vector).sum())
        qweights = dict(zip(vector.indices, vector.data))
        details = []
        for term in sorted(set(preprocess(query)) & self.vocabulary.keys()):
            col = self.vocabulary[term]
            f = float(self.tf[p, col])
            if f:
                bm25 = self.bm25_idf[col] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.doc_lengths[p] / self.avgdl))
                cosine = float(self.tfidf[p, col]) * qweights[col] / (self.doc_norms[p] * qnorm)
                details.append({'term': term, 'tf': int(f), 'df': int(self.df[col]),
                    'length': int(self.doc_lengths[p]), 'tfidf_contribution': float(cosine),
                    'bm25_contribution': float(bm25)})
        return details

    def save(self, directory=INDEX_DIR):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / 'tf.npz', self.tf)
        metadata = {'documents': self.documents, 'terms': list(self.vocabulary), 'k1': self.k1, 'b': self.b}
        (directory / 'index.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, directory=INDEX_DIR):
        directory = Path(directory)
        metadata = json.loads((directory / 'index.json').read_text(encoding='utf-8'))
        return cls(metadata['documents'], metadata['k1'], metadata['b'],
                   terms=metadata['terms'], tf=sparse.load_npz(directory / 'tf.npz'))


if __name__ == '__main__':
    index = SearchIndex(load_documents())
    index.save()
    print(f'Índice guardado: {index.N} documentos, {len(index.vocabulary)} términos, avgdl={index.avgdl:.2f}')
