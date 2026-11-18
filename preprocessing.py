"""Procesamiento idéntico para documentos y consultas; no descarga recursos NLTK."""
from functools import lru_cache
import re
import unicodedata

from nltk.stem.snowball import SnowballStemmer

# Lista explícita y reproducible. Se conservan no, nor, not, without y never.
STOPWORDS = frozenset("""
a an the and or but if while as at by for from in into of on onto to with
about above after again against all am are be because been before being below
between both can could did do does doing down during each few further had has
have having he her here hers herself him himself his how i me my myself is it
its itself just more most other our ours ourselves out over own same she should
so some such than that their theirs them themselves then there these they this
those through too under until up very was we were what when where which who
whom why will would you your yours yourself yourselves s t d ll m re ve
""".split())
STEMMER = SnowballStemmer("english")
# NFKC unifica variantes de ancho y micro; se conservan letras griegas,
# números decimales, identificadores alfanuméricos y comparadores clínicos.
TOKEN_RE = re.compile(r"[−-][0-9]+(?:[.,][0-9]+)*|[^\W_]+(?:[.,][0-9]+)*|<=|>=|[<>=%+−±]", re.UNICODE)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().replace("’", "'")


def tokenize(text: str) -> list[str]:
    text = re.sub(r"n['’]t\b", " not", normalize(text))
    return TOKEN_RE.findall(text)


@lru_cache(maxsize=100_000)
def stem(token: str) -> str:
    return STEMMER.stem(token)


def preprocess(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]

