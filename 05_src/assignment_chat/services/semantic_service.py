import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from services.tfidf_embedder import TFIDFEmbedder


class SemanticService:
    def __init__(self, persist_dir: str, collection_name: str, llm):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.llm = llm

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(allow_reset=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

        tfidf_path = os.path.join(persist_dir, "tfidf_model.json")
        if not os.path.exists(tfidf_path):
            raise RuntimeError(
                f"Missing TF-IDF model at {tfidf_path}. Run scripts/build_embeddings.py first."
            )
        with open(tfidf_path, "r", encoding="utf-8") as f:
            self.embedder = TFIDFEmbedder.from_json(f.read())

    def answer(self, user_text: str, mem_state: Dict[str, Any]) -> str:
        q = (user_text or "").strip()

        # allow "docs:" prefix but also work without it
        for prefix in ["docs:", "search docs:"]:
            if q.lower().startswith(prefix):
                q = q[len(prefix):].strip()

        if not q:
            return "Ask like: `docs: create a 7-day meal plan for weight loss`"

        q_vec = self.embedder.embed(q)

        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=5,
            include=["documents", "metadatas"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        # Fallback: if nothing retrieved, use normal chat
        if not docs:
            return self.llm.chat(mem_state, user_text)

        contexts: List[str] = []
        sources: List[str] = []
        for d, m in zip(docs, metas):
            contexts.append(d)
            sources.append(m.get("source", "unknown"))

        answer = self.llm.synthesize_from_context(mem_state, q, contexts)
        unique_sources = ", ".join(sorted(set(sources)))
        return f"{answer}\n\nSources: {unique_sources}"