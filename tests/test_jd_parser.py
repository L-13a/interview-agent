import pytest
import json
from unittest.mock import patch, MagicMock
from agents.jd_parser import parse_jd


def _mock_deepseek(response_text: str):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=response_text))]
    return mock_resp


def test_parse_jd_returns_required_fields():
    payload = {
        "company": "字节跳动",
        "role": "AI产品经理",
        "role_type": "ai_product",
        "key_skills": ["大模型", "用户增长", "数据分析"]
    }
    with patch("agents.jd_parser._client.chat.completions.create", return_value=_mock_deepseek(json.dumps(payload))):
        result = parse_jd("ByteDance is hiring an AI PM...")
    assert result["company"] == "字节跳动"
    assert result["role_type"] == "ai_product"
    assert isinstance(result["key_skills"], list)
    assert len(result["key_skills"]) >= 1


def test_parse_jd_raises_on_invalid_json():
    with patch("agents.jd_parser._client.chat.completions.create", return_value=_mock_deepseek("not json")):
        with pytest.raises(Exception):
            parse_jd("some jd text")
