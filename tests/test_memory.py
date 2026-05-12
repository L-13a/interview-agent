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
