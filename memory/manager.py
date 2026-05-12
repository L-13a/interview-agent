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
