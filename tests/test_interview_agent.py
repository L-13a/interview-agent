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
