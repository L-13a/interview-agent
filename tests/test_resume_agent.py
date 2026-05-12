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
