import os
import anthropic
from tavily import TavilyClient

_anthropic = anthropic.Anthropic()
_tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "placeholder"))


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
        f"{company} 业务线 产品矩阵 2026",
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
