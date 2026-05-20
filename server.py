import json
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).parent
HISTORY_FILE = BASE / "memory" / "history.json"

PROFILE_FILES = {
    "self": BASE / "knowledge" / "self_profile.md",
    "resume": BASE / "knowledge" / "resume_base.md",
    "experience": BASE / "knowledge" / "experiences" / "experience.md",
}

app = FastAPI(title="Interview Prep Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProfileBody(BaseModel):
    content: str


class HistoryEntry(BaseModel):
    id: int
    date: str
    jd: str
    company: str
    role: str
    role_type: str
    key_skills: List[str]
    resume: str
    research: str
    interview: str


def _read_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_history(entries: list) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@app.get("/api/profile/{key}")
def get_profile(key: str):
    path = PROFILE_FILES.get(key)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown profile key")
    if not path.exists():
        return {"content": ""}
    return {"content": path.read_text(encoding="utf-8")}


@app.put("/api/profile/{key}")
def update_profile(key: str, body: ProfileBody):
    path = PROFILE_FILES.get(key)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown profile key")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.content, encoding="utf-8")
    return {"ok": True}


@app.get("/api/history")
def get_history():
    return {"entries": _read_history()}


@app.post("/api/history")
def save_history_entry(entry: HistoryEntry):
    entries = _read_history()
    entry_dict = entry.model_dump()
    idx = next(
        (i for i, e in enumerate(entries)
         if e.get("jd") == entry.jd and e.get("company") == entry.company),
        -1,
    )
    if idx >= 0:
        entries[idx] = entry_dict
    else:
        entries.insert(0, entry_dict)
    _write_history(entries[:20])
    return {"ok": True}


@app.delete("/api/history/{entry_id}")
def delete_history_entry(entry_id: int):
    entries = _read_history()
    entries = [e for e in entries if e.get("id") != entry_id]
    _write_history(entries)
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(BASE / "frontend"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
