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


def test_research_returns_string():
    with patch("agents.research_agent._tavily", _mock_tavily("字节跳动旗下有抖音、今日头条")), \
         patch("agents.research_agent._anthropic.messages.create", return_value=_mock_claude("## 核心业务线\n- 抖音")):
        result = run(JD_INFO)
    assert isinstance(result, str)
    assert len(result) > 0


def test_research_makes_three_searches():
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
