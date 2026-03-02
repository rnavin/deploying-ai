import json
import math
import re
from collections import Counter
from typing import Dict, List

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]

class TFIDFEmbedder:
    """
    Minimal TF-IDF embedder (no external deps).
    Stores vocab -> index and idf values.
    Produces L2-normalized dense vectors (list[float]).
    """

    def __init__(self, vocab: Dict[str, int], idf: List[float]):
        self.vocab = vocab
        self.idf = idf
        self.dim = len(idf)

    @staticmethod
    def fit(texts: List[str], min_df: int = 1, max_features: int = 20000) -> "TFIDFEmbedder":
        df = Counter()
        for txt in texts:
            toks = set(tokenize(txt))
            for t in toks:
                df[t] += 1

        items = [(t, c) for t, c in df.items() if c >= min_df]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[:max_features]

        vocab = {t: i for i, (t, _) in enumerate(items)}
        n_docs = max(1, len(texts))

        idf = [0.0] * len(vocab)
        for t, i in vocab.items():
            idf[i] = math.log((1 + n_docs) / (1 + df[t])) + 1.0

        return TFIDFEmbedder(vocab=vocab, idf=idf)

    def embed(self, text: str) -> List[float]:
        toks = tokenize(text)
        if not toks or self.dim == 0:
            return [0.0] * self.dim

        tf = Counter(toks)
        vec = [0.0] * self.dim
        for term, cnt in tf.items():
            idx = self.vocab.get(term)
            if idx is None:
                continue
            vec[idx] = cnt * self.idf[idx]

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def to_json(self) -> str:
        return json.dumps({"vocab": self.vocab, "idf": self.idf})

    @staticmethod
    def from_json(s: str) -> "TFIDFEmbedder":
        obj = json.loads(s)
        return TFIDFEmbedder(vocab=obj["vocab"], idf=obj["idf"])