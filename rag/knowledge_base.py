import chromadb
from pathlib import Path


class KnowledgeBase:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("knowledge")

    def index_documents(self, knowledge_dir: str = "./knowledge") -> int:
        chunks, ids, metadatas = [], [], []
        for path in Path(knowledge_dir).rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
            for i, para in enumerate(paragraphs):
                chunk_id = f"{path.stem}_{i}"
                chunks.append(para)
                ids.append(chunk_id)
                metadatas.append({"source": str(path), "chunk": i})
        if chunks:
            self.collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def query(self, query_text: str, n_results: int = 3) -> list[str]:
        if self.collection.count() == 0:
            return []
        actual_n = min(n_results, self.collection.count())
        results = self.collection.query(query_texts=[query_text], n_results=actual_n)
        return results["documents"][0] if results["documents"] else []
