from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from book_see_rag.config import get_settings
from book_see_rag.metadata_store import get_knowledge_base, list_documents


@lru_cache(maxsize=1)
def load_benchmark_profile_map() -> dict[str, dict]:
    path = Path(get_settings().metadata_dir).parent / "benchmark_profiles.json"
    if not path.exists():
        path = Path("data/benchmark_profiles.json")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {}) if isinstance(payload, dict) else {}
    return profiles if isinstance(profiles, dict) else {}


def resolve_doc_profile(filename: str) -> str | None:
    matches = [doc for doc in list_documents() if doc["filename"] == filename]
    if not matches:
        return None
    kb = get_knowledge_base(matches[-1]["kb_id"])
    if not kb:
        return None
    profile = kb.get("query_profile")
    return str(profile) if profile else None


def resolve_eval_set_for_profile(profile_name: str | None) -> str | None:
    if not profile_name:
        return None
    profile = load_benchmark_profile_map().get(profile_name)
    if not profile:
        return None
    eval_set = profile.get("eval_set")
    return str(eval_set) if eval_set else None
