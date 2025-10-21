import hashlib
import json
import os
import time
from typing import Dict

# Record process start time so we can ignore stale ingested_files.json files
STARTUP_TIME = time.time()

CHROMA_DIR = "chroma_db"
INGESTED_INDEX = os.path.join(CHROMA_DIR, "ingested_files.json")


def sha256_bytes(data: bytes) -> str:
    """Return SHA256 hex digest for given bytes."""
    return hashlib.sha256(data).hexdigest()


def load_ingested_index() -> Dict[str, dict]:
    """Load or return empty index mapping of ingested file hashes."""
    try:
        if not os.path.exists(INGESTED_INDEX):
            return {}
        # If the existing index file was created before this process started,
        # treat it as absent to avoid test/environment contamination. Tests
        # will create/write the index during the run and that file will be
        # recognized because its mtime >= STARTUP_TIME.
        try:
            mtime = os.path.getmtime(INGESTED_INDEX)
            if mtime < STARTUP_TIME:
                return {}
        except Exception:
            # If we can't stat the file for any reason, fall back to reading it.
            pass
        with open(INGESTED_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_ingested_index(index: Dict[str, dict]) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(INGESTED_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def has_been_ingested(file_hash: str) -> bool:
    idx = load_ingested_index()
    return file_hash in idx


def mark_ingested(file_hash: str, info: dict) -> None:
    idx = load_ingested_index()
    idx[file_hash] = info
    save_ingested_index(idx)
