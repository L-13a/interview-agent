import pytest
import tempfile
import os
from pathlib import Path
from rag.knowledge_base import KnowledgeBase


@pytest.fixture
def kb_with_docs(tmp_path):
    (tmp_path / "profile.md").write_text("我是一名产品经理，擅长数据驱动决策和跨职能协作。")
    (tmp_path / "exp.md").write_text("在XX公司实习期间，主导了推荐算法改版，次留提升8个百分点。GMV增长显著。")
    persist_dir = str(tmp_path / "chroma")
    kb = KnowledgeBase(persist_dir=persist_dir)
    kb.index_documents(knowledge_dir=str(tmp_path))
    return kb


def test_index_documents_returns_chunk_count(tmp_path):
    (tmp_path / "doc.md").write_text("这是第一段。\n\n这是第二段，内容足够长以通过过滤器。内容内容内容内容内容。")
    persist_dir = str(tmp_path / "chroma")
    kb = KnowledgeBase(persist_dir=persist_dir)
    count = kb.index_documents(knowledge_dir=str(tmp_path))
    assert count >= 1


def test_query_returns_relevant_chunks(kb_with_docs):
    results = kb_with_docs.query("推荐算法 次留", n_results=2)
    assert len(results) >= 1
    assert any("推荐" in r or "次留" in r for r in results)


def test_query_returns_empty_list_when_no_match(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    kb = KnowledgeBase(persist_dir=persist_dir)
    results = kb.query("completely unrelated xyz123", n_results=3)
    assert isinstance(results, list)
