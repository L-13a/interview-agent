# Job Interview Prep Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI agent that takes a JD as input and outputs a tailored resume, company research, and interview questions using multi-agent orchestration, RAG, memory, reflection, and human-in-the-loop.

**Architecture:** Orchestrator parses JD and runs Phase 1 (Resume Agent + Research Agent in parallel via asyncio), pauses for human review of research results, then runs Phase 2 (Interview Prep Agent). Results are saved as a Markdown report and logged to persistent memory.

**Tech Stack:** Python 3.11+, anthropic SDK, chromadb, tavily-python, rich, python-dotenv

---

## File Map

| File | Responsibility |
|---|---|
| `main.py` | Orchestrator: pipeline, asyncio, HITL, report assembly |
| `agents/jd_parser.py` | `parse_jd(jd_text) → dict` via Claude |
| `agents/resume_agent.py` | `run(jd_info, kb) → str` with reflection loop |
| `agents/research_agent.py` | `run(jd_info) → str` via Tavily + Claude |
| `agents/interview_agent.py` | `run(jd_info, company_research) → str` |
| `rag/knowledge_base.py` | `KnowledgeBase`: index docs, query chunks |
| `memory/manager.py` | `load_history`, `save_application`, `find_similar` |
| `knowledge/*.md` | User-maintained experience docs (sample provided) |
| `tests/` | Unit tests for each module |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `agents/__init__.py`, `rag/__init__.py`, `memory/__init__.py`, `tests/__init__.py`
- Create: `knowledge/self_profile.md`, `knowledge/resume_base.md`, `knowledge/experiences/sample_internship.md`

- [ ] **Step 1: Create requirements.txt**

```
anthropic>=0.40.0
chromadb>=0.5.0
tavily-python>=0.3.0
rich>=13.0.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Create .env.example**

```
ANTHROPIC_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

- [ ] **Step 3: Create directory structure and empty __init__.py files**

```bash
mkdir -p agents rag memory knowledge/experiences knowledge/projects outputs tests
touch agents/__init__.py rag/__init__.py memory/__init__.py tests/__init__.py
cp .env.example .env
# fill in real API keys in .env
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error. chromadb will download an embedding model (~60MB) on first use.

- [ ] **Step 5: Create sample knowledge docs**

`knowledge/self_profile.md`:
```markdown
# 个人简介

**姓名：** 刘莉姗
**求职意向：** 产品经理
**核心优势：** 数据驱动、跨职能协作、0-1 产品经验
**技术背景：** 计算机科学本科，熟悉机器学习基础，有 Python 开发经验

## 个人特点
- 有独立搭建过 AI 工具的经历，理解工程与产品的边界
- 习惯用数据验证决策
- 善于将复杂需求拆解为可执行的迭代计划
```

`knowledge/resume_base.md`:
```markdown
# 简历底稿

## 教育背景
哈尔滨工业大学（深圳） 计算机科学与技术 本科 2023-2027

## 实习经历

### YouWare 产品经理实习生 2026.01-2026.05
- 负责 XX 功能需求调研与 PRD 撰写
- 协调研发、设计、运营三方完成功能迭代上线
- 上线后 DAU 提升 XX%，用户反馈好评率 XX%


## 项目经历

### AI 求职助手（个人项目）2026.04-至今
- 独立设计并开发基于 Claude API 的多 Agent 求职辅助工具
- 实现 RAG、多 Agent 协作、Reflection、Human-in-the-loop 等核心 Agent 技术
- 技术栈：Python、ChromaDB、Tavily API、Anthropic SDK
```

`knowledge/experiences/sample_internship.md`:
```markdown
# XX 公司产品实习详细经历

## 背景
Youware是一家做Coding Agent的公司，我在产品部门参与主产品Youware的功能迭代与一些子产品落地、推广。

## 具体工作内容

### 功能 A：用户反馈分类系统
- 问题：每天有 500+ 条用户反馈，人工分类效率低
- 方案：设计半自动分类流程，引入关键词规则 + 人工复核
- 结果：分类效率提升 60%，响应时效从 3 天缩短到 1 天

### 功能 B：推荐算法改版需求文档
- 协作研发和算法团队，撰写推荐改版 PRD
- 通过用研访谈（共访谈 15 名用户）确认核心需求
- 改版上线后次留提升 8 个百分点

## 学到的东西
- 如何在资源有限的情况下推动项目落地
- 数据埋点设计与 A/B 实验方案制定
- 跨部门沟通和优先级排期的实际经验
```

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt .env.example agents/ rag/ memory/ tests/ knowledge/
git commit -m "feat: project scaffold and sample knowledge docs"
```

---

## Task 2: KnowledgeBase (RAG)

**Files:**
- Create: `rag/knowledge_base.py`
- Create: `tests/test_knowledge_base.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_knowledge_base.py`:
```python
import pytest
import tempfile
import os
from pathlib import Path
from rag.knowledge_base import KnowledgeBase


@pytest.fixture
def kb_with_docs(tmp_path):
    # Create sample knowledge files
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_knowledge_base.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` (file doesn't exist yet)

- [ ] **Step 3: Implement KnowledgeBase**

`rag/knowledge_base.py`:
```python
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
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_knowledge_base.py -v
```
Expected: all 4 tests PASS. First run downloads the embedding model (~60MB), takes ~30s.

- [ ] **Step 5: Commit**

```bash
git add rag/knowledge_base.py tests/test_knowledge_base.py
git commit -m "feat: RAG knowledge base with ChromaDB"
```

---

## Task 3: Memory Manager

**Files:**
- Create: `memory/manager.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_memory.py`:
```python
import pytest
import json
from pathlib import Path
from unittest.mock import patch
from memory.manager import load_history, save_application, find_similar


def test_load_history_returns_empty_when_no_file(tmp_path):
    history_path = tmp_path / "history.json"
    with patch("memory.manager.HISTORY_PATH", history_path):
        result = load_history()
    assert result == {"applications": []}


def test_save_and_load_application(tmp_path):
    history_path = tmp_path / "history.json"
    with patch("memory.manager.HISTORY_PATH", history_path):
        save_application(
            company="字节跳动",
            role="AI产品经理",
            role_type="ai_product",
            jd_keywords=["大模型", "C端"],
            output_path="outputs/test.md"
        )
        history = load_history()
    assert len(history["applications"]) == 1
    assert history["applications"][0]["company"] == "字节跳动"


def test_find_similar_matches_same_company(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps({
        "applications": [
            {"company": "字节跳动", "role": "AI产品经理", "role_type": "ai_product",
             "jd_keywords": [], "output_path": "", "date": "2026-01-01"}
        ]
    }))
    with patch("memory.manager.HISTORY_PATH", history_path):
        results = find_similar(company="字节跳动", role_type="commercial")
    assert len(results) == 1


def test_find_similar_returns_empty_for_new_company(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps({"applications": []}))
    with patch("memory.manager.HISTORY_PATH", history_path):
        results = find_similar(company="新公司", role_type="ai_product")
    assert results == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_memory.py -v
```
Expected: `ImportError` (file doesn't exist yet)

- [ ] **Step 3: Implement memory manager**

`memory/manager.py`:
```python
import json
from datetime import date
from pathlib import Path

HISTORY_PATH = Path("./memory/history.json")


def load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {"applications": []}
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"applications": []}


def save_application(
    company: str,
    role: str,
    role_type: str,
    jd_keywords: list[str],
    output_path: str,
) -> None:
    history = load_history()
    history["applications"].append({
        "date": str(date.today()),
        "company": company,
        "role": role,
        "role_type": role_type,
        "jd_keywords": jd_keywords,
        "output_path": output_path,
    })
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def find_similar(company: str, role_type: str) -> list[dict]:
    history = load_history()
    return [
        app for app in history["applications"]
        if app.get("company") == company or app.get("role_type") == role_type
    ]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_memory.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add memory/manager.py tests/test_memory.py
git commit -m "feat: persistent memory manager for application history"
```

---

## Task 4: JD Parser

**Files:**
- Create: `agents/jd_parser.py`
- Create: `tests/test_jd_parser.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_jd_parser.py`:
```python
import pytest
import json
from unittest.mock import patch, MagicMock
from agents.jd_parser import parse_jd


def _mock_claude(response_text: str):
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=response_text)]
    return mock_resp


def test_parse_jd_returns_required_fields():
    payload = {
        "company": "字节跳动",
        "role": "AI产品经理",
        "role_type": "ai_product",
        "key_skills": ["大模型", "用户增长", "数据分析"]
    }
    with patch("agents.jd_parser._client.messages.create", return_value=_mock_claude(json.dumps(payload))):
        result = parse_jd("ByteDance is hiring an AI PM...")
    assert result["company"] == "字节跳动"
    assert result["role_type"] == "ai_product"
    assert isinstance(result["key_skills"], list)
    assert len(result["key_skills"]) >= 1


def test_parse_jd_raises_on_invalid_json():
    with patch("agents.jd_parser._client.messages.create", return_value=_mock_claude("not json")):
        with pytest.raises(Exception):
            parse_jd("some jd text")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_jd_parser.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement JD parser**

`agents/jd_parser.py`:
```python
import json
import anthropic

_client = anthropic.Anthropic()


def parse_jd(jd_text: str) -> dict:
    """Parse JD text into structured dict.

    Returns:
        {"company": str, "role": str, "role_type": str, "key_skills": list[str]}
        role_type is one of: commercial, ai_product, growth, platform, other
    """
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="You extract structured information from job descriptions. Return only valid JSON, no explanation.",
        messages=[{
            "role": "user",
            "content": f"""Extract from this job description:
- company: company name (string)
- role: exact job title (string)
- role_type: one of "commercial", "ai_product", "growth", "platform", "other"
- key_skills: list of 3-5 core required skills (strings, in Chinese if the JD is Chinese)

JD:
{jd_text}"""
        }]
    )
    return json.loads(response.content[0].text)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_jd_parser.py -v
```
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add agents/jd_parser.py tests/test_jd_parser.py
git commit -m "feat: JD parser using Claude structured output"
```

---

## Task 5: Research Agent

**Files:**
- Create: `agents/research_agent.py`
- Create: `tests/test_research_agent.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_research_agent.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from agents.research_agent import run


JD_INFO = {
    "company": "字节跳动",
    "role": "AI产品经理",
    "role_type": "ai_product",
    "key_skills": ["大模型", "C端产品"]
}


def _mock_tavily(content: str):
    mock = MagicMock()
    mock.search.return_value = {"results": [{"content": content}]}
    return mock


def _mock_claude(text: str):
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=text)]
    return mock_resp


def test_research_returns_string(tmp_path):
    with patch("agents.research_agent._tavily", _mock_tavily("字节跳动旗下有抖音、今日头条")), \
         patch("agents.research_agent._anthropic.messages.create", return_value=_mock_claude("## 核心业务线\n- 抖音")):
        result = run(JD_INFO)
    assert isinstance(result, str)
    assert len(result) > 0


def test_research_makes_three_searches(tmp_path):
    mock_tavily = _mock_tavily("some content")
    with patch("agents.research_agent._tavily", mock_tavily), \
         patch("agents.research_agent._anthropic.messages.create", return_value=_mock_claude("result")):
        run(JD_INFO)
    assert mock_tavily.search.call_count == 3


def test_research_degrades_gracefully_on_tavily_failure():
    mock_tavily = MagicMock()
    mock_tavily.search.side_effect = Exception("API error")
    with patch("agents.research_agent._tavily", mock_tavily), \
         patch("agents.research_agent._anthropic.messages.create", return_value=_mock_claude("Fallback analysis")):
        result = run(JD_INFO)
    assert isinstance(result, str)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_research_agent.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement research agent**

`agents/research_agent.py`:
```python
import os
import anthropic
from tavily import TavilyClient

_anthropic = anthropic.Anthropic()
_tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))


def run(jd_info: dict) -> str:
    """Research company and return formatted markdown summary."""
    company = jd_info["company"]
    role = jd_info["role"]

    search_results = _gather_search_results(company, jd_info.get("role_type", ""))

    response = _anthropic.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Based on the following information about {company}, write a structured company research summary for a candidate applying for {role}.

Include these sections in Chinese markdown:
## 核心业务线（3-5条，附简要说明）
## 近期动态（1-2条最新消息）
## 公司文化关键词
## What You Should Know（3条面试加分提示）

Source information:
{search_results}"""
        }]
    )
    return response.content[0].text


def _gather_search_results(company: str, role_type: str) -> str:
    queries = [
        f"{company} 业务线 产品矩阵 2025",
        f"{company} {role_type} 战略 最新动态",
        f"{company} 公司文化 价值观 面试风格",
    ]
    snippets = []
    for q in queries:
        try:
            results = _tavily.search(query=q, max_results=3)
            snippets.extend(r["content"] for r in results.get("results", []))
        except Exception:
            pass
    return "\n\n".join(snippets[:9]) if snippets else f"Direct analysis of {company} based on general knowledge."
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_research_agent.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agents/research_agent.py tests/test_research_agent.py
git commit -m "feat: research agent with Tavily search and graceful degradation"
```

---

## Task 6: Resume Agent (with Reflection)

**Files:**
- Create: `agents/resume_agent.py`
- Create: `tests/test_resume_agent.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_resume_agent.py`:
```python
import pytest
from unittest.mock import patch, MagicMock, call
from agents.resume_agent import run


JD_INFO = {
    "company": "字节跳动",
    "role": "AI产品经理",
    "role_type": "ai_product",
    "key_skills": ["大模型", "用户增长"]
}


def _mock_kb(chunks):
    kb = MagicMock()
    kb.query.return_value = chunks
    return kb


def _claude_seq(*texts):
    """Return successive mock responses for sequential Claude calls."""
    responses = []
    for text in texts:
        mock = MagicMock()
        mock.content = [MagicMock(text=text)]
        responses.append(mock)
    return responses


def test_resume_agent_returns_string():
    kb = _mock_kb(["负责推荐算法改版，次留提升8%"])
    with patch("agents.resume_agent._client.messages.create",
               side_effect=_claude_seq("初稿bullets", "critique结果", "最终bullets")):
        result = run(JD_INFO, kb)
    assert isinstance(result, str)
    assert len(result) > 0


def test_resume_agent_calls_claude_three_times():
    kb = _mock_kb(["some experience"])
    with patch("agents.resume_agent._client.messages.create",
               side_effect=_claude_seq("draft", "critique", "final")) as mock_create:
        run(JD_INFO, kb)
    assert mock_create.call_count == 3


def test_resume_agent_warns_when_kb_empty():
    kb = _mock_kb([])
    result = run(JD_INFO, kb)
    assert "未找到相关经历" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_resume_agent.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement resume agent**

`agents/resume_agent.py`:
```python
import anthropic

_client = anthropic.Anthropic()


def run(jd_info: dict, kb) -> str:
    """Generate tailored resume bullets with a reflection loop."""
    chunks = kb.query(" ".join(jd_info["key_skills"]) + " " + jd_info["role_type"], n_results=3)

    if not chunks:
        return "⚠️ 未找到相关经历，请在 knowledge/ 目录下补充你的实习和项目经历文档。"

    context = "\n\n---\n\n".join(chunks)
    draft = _generate_bullets(jd_info, context)
    critique = _critique_bullets(jd_info, draft)
    return _revise_bullets(jd_info, draft, critique)


def _generate_bullets(jd_info: dict, context: str) -> str:
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""你是专业简历顾问。根据候选人经历，为以下岗位撰写定制简历 bullets。

目标岗位：{jd_info['role']}（{jd_info['company']}）
岗位类型：{jd_info['role_type']}
核心要求：{', '.join(jd_info['key_skills'])}

候选人相关经历：
{context}

每段经历写 2-3 条 bullet，要求：
- 动词开头
- 优先包含量化数据
- 突出与目标岗位相关的能力

用中文输出。"""
        }]
    )
    return response.content[0].text


def _critique_bullets(jd_info: dict, draft: str) -> str:
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""审查以下简历 bullets，目标岗位要求：{', '.join(jd_info['key_skills'])}。

{draft}

对每条 bullet 检查：
1. 是否体现了岗位核心要求中的某项能力？
2. 是否有量化数据？
3. 是否动词开头？

列出需要改进的具体建议，简短。"""
        }]
    )
    return response.content[0].text


def _revise_bullets(jd_info: dict, draft: str, critique: str) -> str:
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""根据以下审查意见，修改简历 bullets。只输出修改后的 bullets，不需要解释。

原始 bullets：
{draft}

审查意见：
{critique}

目标岗位：{jd_info['role']}（{jd_info['company']}）
用中文输出。"""
        }]
    )
    return response.content[0].text
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_resume_agent.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agents/resume_agent.py tests/test_resume_agent.py
git commit -m "feat: resume agent with three-step reflection loop"
```

---

## Task 7: Interview Prep Agent

**Files:**
- Create: `agents/interview_agent.py`
- Create: `tests/test_interview_agent.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_interview_agent.py`:
```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from agents.interview_agent import run


JD_INFO = {
    "company": "字节跳动",
    "role": "AI产品经理",
    "role_type": "ai_product",
    "key_skills": ["大模型", "用户增长"]
}
COMPANY_RESEARCH = "## 核心业务线\n- 抖音\n- 今日头条\n\n## 公司文化\n字节跳动注重数据驱动。"


def _mock_claude(text: str):
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    return mock


def test_interview_agent_returns_string(tmp_path):
    profile = tmp_path / "self_profile.md"
    profile.write_text("我是AI方向的产品经理候选人。")
    with patch("agents.interview_agent.PROFILE_PATH", profile), \
         patch("agents.interview_agent._client.messages.create",
               return_value=_mock_claude("## 行为题\n1. 请描述一次...")):
        result = run(JD_INFO, COMPANY_RESEARCH)
    assert isinstance(result, str)
    assert len(result) > 0


def test_interview_agent_works_without_profile(tmp_path):
    nonexistent = tmp_path / "missing.md"
    with patch("agents.interview_agent.PROFILE_PATH", nonexistent), \
         patch("agents.interview_agent._client.messages.create",
               return_value=_mock_claude("## 面试题\n1. ...")):
        result = run(JD_INFO, COMPANY_RESEARCH)
    assert isinstance(result, str)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_interview_agent.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement interview prep agent**

`agents/interview_agent.py`:
```python
import anthropic
from pathlib import Path

_client = anthropic.Anthropic()
PROFILE_PATH = Path("./knowledge/self_profile.md")


def run(jd_info: dict, company_research: str) -> str:
    """Generate interview question bank with answer framework hints."""
    profile = PROFILE_PATH.read_text(encoding="utf-8") if PROFILE_PATH.exists() else "（无个人简介）"

    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""为以下候选人生成面试准备题库。

目标公司：{jd_info['company']}
目标岗位：{jd_info['role']}
核心技能要求：{', '.join(jd_info['key_skills'])}

公司研究资料：
{company_research}

候选人背景：
{profile}

请生成以下四类题目（用中文 Markdown 格式）：

## 行为题（3条）
结合候选人背景，STAR 格式。每题附一行**回答框架**提示。

## 产品题（3条）
结合该公司具体产品，例如"如何提升XX的次日留存"。每题附一行**回答框架**提示。

## 技术认知题（2条）
AI PM 专项：模型评估、AI功能成功指标定义等。每题附一行**回答框架**提示。

## 反问题（2条）
建议候选人向面试官提问的高质量问题（体现产品思维和对公司的了解）。"""
        }]
    )
    return response.content[0].text
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_interview_agent.py -v
```
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add agents/interview_agent.py tests/test_interview_agent.py
git commit -m "feat: interview prep agent with question bank and answer frameworks"
```

---

## Task 8: Main Pipeline

**Files:**
- Create: `main.py`

- [ ] **Step 1: Implement main.py**

`main.py`:
```python
import asyncio
import os
import sys
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from rag.knowledge_base import KnowledgeBase
from memory.manager import load_history, save_application, find_similar
from agents.jd_parser import parse_jd
from agents.resume_agent import run as resume_run
from agents.research_agent import run as research_run
from agents.interview_agent import run as interview_run

load_dotenv()
console = Console()


def _read_jd() -> str:
    console.print("\n[bold cyan]请粘贴 JD 内容，粘贴完成后按两次 Enter：[/bold cyan]")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines[:-1]).strip()


async def _run_phase1(jd_info: dict, kb: KnowledgeBase) -> tuple[str, str]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=2) as pool:
        resume_future = loop.run_in_executor(pool, resume_run, jd_info, kb)
        research_future = loop.run_in_executor(pool, research_run, jd_info)
        resume_result, research_result = await asyncio.gather(resume_future, research_future)
    return resume_result, research_result


def main():
    # ── Setup ──────────────────────────────────────────────
    console.print(Panel("[bold]Job Interview Prep Agent[/bold]", border_style="cyan"))

    console.print("\n[dim]正在初始化知识库...[/dim]")
    kb = KnowledgeBase()
    count = kb.index_documents()
    console.print(f"[green]✓ 知识库就绪，共 {count} 个文本块[/green]")

    # ── JD Input & Parse ───────────────────────────────────
    jd_text = _read_jd()
    if not jd_text:
        console.print("[red]JD 内容为空，退出。[/red]")
        sys.exit(1)

    console.print("\n[dim]解析 JD 中...[/dim]")
    jd_info = parse_jd(jd_text)
    console.print(
        f"[green]✓ 公司：{jd_info['company']} | 岗位：{jd_info['role']} | 类型：{jd_info['role_type']}[/green]"
    )

    # ── Memory Check ───────────────────────────────────────
    similar = find_similar(jd_info["company"], jd_info["role_type"])
    if similar:
        names = "、".join(f"{a['company']} {a['role']}" for a in similar[:3])
        console.print(f"[yellow]💡 发现历史投递记录：{names}[/yellow]")

    # ── Phase 1: Parallel ──────────────────────────────────
    console.print("\n[bold]Phase 1：简历改写 & 公司研究（并行执行中）...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
        transient=True,
    ) as progress:
        t1 = progress.add_task("Resume Agent  — RAG 召回 + 生成 + Reflection", total=None)
        t2 = progress.add_task("Research Agent — Tavily 搜索 + 汇总", total=None)
        resume_result, research_result = asyncio.run(_run_phase1(jd_info, kb))
        progress.update(t1, completed=True)
        progress.update(t2, completed=True)

    console.print("[green]✓ Phase 1 完成[/green]")

    # ── Human-in-the-Loop ──────────────────────────────────
    console.print("\n[bold cyan]⏸ 公司研究结果如下，请确认：[/bold cyan]")
    console.print(Markdown(research_result))
    console.print("\n[yellow]如有补充或纠正请输入，直接回车跳过：[/yellow]", end="")
    correction = input().strip()
    if correction:
        research_result += f"\n\n**补充信息（用户提供）：** {correction}"

    # ── Phase 2 ────────────────────────────────────────────
    console.print("\n[bold]Phase 2：生成面试题库...[/bold]")
    interview_result = interview_run(jd_info, research_result)
    console.print("[green]✓ Phase 2 完成[/green]")

    # ── Assemble Report ────────────────────────────────────
    today = date.today().strftime("%Y%m%d")
    output_path = f"outputs/{jd_info['company']}_{today}.md"
    Path("outputs").mkdir(exist_ok=True)

    report = f"""# 求职准备报告 — {jd_info['company']} {jd_info['role']}

**生成日期：** {date.today()}

---

## 一、定制简历

{resume_result}

---

## 二、公司研究

{research_result}

---

## 三、面试题库

{interview_result}
"""
    Path(output_path).write_text(report, encoding="utf-8")

    # ── Display & Save ─────────────────────────────────────
    console.print("\n")
    console.print(Panel(Markdown(report), title=f"✅ 报告完成 → {output_path}", border_style="green"))

    save_application(
        company=jd_info["company"],
        role=jd_info["role"],
        role_type=jd_info["role_type"],
        jd_keywords=jd_info["key_skills"],
        output_path=output_path,
    )
    console.print(f"\n[dim]已记录到 memory/history.json[/dim]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests to confirm nothing is broken**

```bash
pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: main pipeline with asyncio phase1, HITL, and rich output"
```

---

## Task 9: Smoke Test (End-to-End)

**Files:**
- No new files — runs the actual pipeline

- [ ] **Step 1: Fill in your real knowledge docs**

Edit these files with your actual experience before running:
- `knowledge/self_profile.md` — your real background and strengths
- `knowledge/resume_base.md` — your actual internship/project experience
- `knowledge/experiences/sample_internship.md` — detailed version of one internship

- [ ] **Step 2: Set your API keys**

```bash
# Edit .env and fill in real keys:
# ANTHROPIC_API_KEY=sk-ant-...
# TAVILY_API_KEY=tvly-...
```

- [ ] **Step 3: Run with a real JD**

```bash
python main.py
```

Paste a real JD you want to apply for. Press Enter twice when done.

Expected output sequence:
```
✓ 知识库就绪，共 N 个文本块
✓ 公司：XX | 岗位：AI产品经理 | 类型：ai_product
Phase 1：[两个进度条同时转动，约 20-30 秒]
✓ Phase 1 完成
⏸ 公司研究结果如下，请确认：
[显示公司研究 Markdown]
如有补充或纠正请输入，直接回车跳过：
Phase 2：生成面试题库...
✓ Phase 2 完成
[完整报告渲染]
✓ 报告完成 → outputs/XX_20260511.md
已记录到 memory/history.json
```

- [ ] **Step 4: Verify report file exists and is readable**

```bash
ls -la outputs/
cat outputs/*.md | head -50
```

Expected: Markdown file with three sections — 定制简历, 公司研究, 面试题库

- [ ] **Step 5: Run a second JD to verify memory works**

Run `python main.py` again with a different JD from the same company or same role type.

Expected: terminal shows `💡 发现历史投递记录：...` before Phase 1.

- [ ] **Step 6: Final commit**

```bash
git add knowledge/ memory/ outputs/.gitkeep
git commit -m "feat: complete job interview prep agent — live demo ready"
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single module
pytest tests/test_knowledge_base.py -v
pytest tests/test_memory.py -v
pytest tests/test_jd_parser.py -v
pytest tests/test_resume_agent.py -v
pytest tests/test_research_agent.py -v
pytest tests/test_interview_agent.py -v
```

## Demo Cheat Sheet

```
python main.py
→ paste JD → Enter Enter
→ watch Phase 1 spinners (~25s)
→ review research → Enter to confirm
→ watch Phase 2 (~20s)
→ point at report sections for interviewer
```
