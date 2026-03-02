import sys
import os

# Path fix so imports work when running directly
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # 05_src/assignment_chat
sys.path.append(PROJECT_ROOT)

import glob
import uuid
import re
from typing import List, Iterator

import chromadb
from chromadb.config import Settings

from utils_secrets import apply_secrets_to_env
from services.tfidf_embedder import TFIDFEmbedder


_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

def strip_fenced_code(text: str) -> str:
    return _CODE_FENCE_RE.sub(" ", text)

def is_code_heavy(text: str) -> bool:
    t = text.strip()
    if len(t) < 200:
        return False
    code_tokens = sum(t.count(x) for x in ["def ", "import ", "class ", "return ", "==", "->", "{", "}", "();", "   "])
    lines = t.splitlines()
    indented = sum(1 for ln in lines if ln.startswith("    ") or ln.startswith("\t"))
    if code_tokens >= 12:
        return True
    if lines and (indented / max(1, len(lines))) > 0.35:
        return True
    return False

def stream_chunks_from_file(
    filepath: str,
    chunk_size: int = 1100,
    overlap: int = 200,
    max_chars_per_file: int = 2_000_000,
    encoding: str = "utf-8",
) -> Iterator[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    buf = ""
    total_read = 0
    with open(filepath, "r", encoding=encoding, errors="ignore") as f:
        while True:
            block = f.read(8192)
            if not block:
                break

            remaining = max_chars_per_file - total_read
            if remaining <= 0:
                break

            if len(block) > remaining:
                block = block[:remaining]

            total_read += len(block)
            buf += block

            while len(buf) >= chunk_size:
                chunk = buf[:chunk_size].strip()
                if chunk:
                    yield chunk
                buf = buf[chunk_size - overlap :]

    tail = buf.strip()
    if tail:
        yield tail

def main():
    # load env (not strictly needed for TF-IDF, but consistent)
    secrets_path = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".secrets"))
    apply_secrets_to_env(secrets_path)

    kb_dir = os.path.join(PROJECT_ROOT, "data", "kb")
    persist_dir = os.path.join(PROJECT_ROOT, "data", "chroma_store")
    os.makedirs(persist_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(kb_dir, "*.txt")))
    if not files:
        print(f"No .txt files found in {kb_dir}. Put your files there and re-run.")
        return

    docs: List[str] = []
    metas: List[dict] = []

    for fp in files:
        source = os.path.basename(fp)
        kept = 0
        skipped = 0
        chunk_idx = 0

        for chunk in stream_chunks_from_file(fp):
            chunk_clean = strip_fenced_code(chunk).strip()
            if not chunk_clean or is_code_heavy(chunk_clean):
                skipped += 1
                continue

            docs.append(chunk_clean)
            metas.append({"source": source, "chunk": chunk_idx})
            chunk_idx += 1
            kept += 1

        print(f"Prepared: {source} | kept: {kept} | skipped(code/empty): {skipped}")

    if not docs:
        print("No usable chunks produced after filtering. Check your files.")
        return

    embedder = TFIDFEmbedder.fit(docs, min_df=1, max_features=20000)
    embeddings = [embedder.embed(d) for d in docs]

    tfidf_path = os.path.join(persist_dir, "tfidf_model.json")
    with open(tfidf_path, "w", encoding="utf-8") as f:
        f.write(embedder.to_json())

    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    client.reset()
    collection = client.get_or_create_collection(name="assignment_kb")

    ids = [str(uuid.uuid4()) for _ in docs]
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    print(f"\n Built ChromaDB index from {len(files)} files")
    print(f" Total chunks stored: {len(docs)}")
    print(f" Chroma stored at: {persist_dir}")
    print(f" TF-IDF model saved at: {tfidf_path}")

if __name__ == "__main__":
    main()