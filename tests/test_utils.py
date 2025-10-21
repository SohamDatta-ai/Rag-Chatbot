import json
import os
import shutil

from utils import (
    CHROMA_DIR,
    has_been_ingested,
    load_ingested_index,
    mark_ingested,
    sha256_bytes,
)


def test_sha256_bytes():
    data = b"hello"
    h = sha256_bytes(data)
    assert isinstance(h, str)
    assert len(h) == 64


def test_ingest_index_lifecycle(tmp_path):
    # Ensure clean chroma_dir for the test
    test_dir = tmp_path / "chroma_db"
    test_dir.mkdir()

    # Point at test chroma dir by creating file path
    ingested_path = test_dir / "ingested_files.json"

    # Ensure module functions write to this test path by monkeypatching environment
    # Since utils uses CHROMA_DIR constant, we'll simulate by creating the file directly
    try:
        # Initially no index
        assert load_ingested_index() == {}

        sample_hash = sha256_bytes(b"sample")
        assert not has_been_ingested(sample_hash)

        # mark ingested by writing directly to the index file path used by utils
        # Create chroma_db folder and write file
        real_dir = CHROMA_DIR
        os.makedirs(real_dir, exist_ok=True)
        real_index = os.path.join(real_dir, "ingested_files.json")
        with open(real_index, "w", encoding="utf-8") as f:
            json.dump({}, f)

        # Now use mark_ingested
        mark_ingested(sample_hash, {"time": 0, "chunks": 1})
        idx = load_ingested_index()
        assert sample_hash in idx
        assert has_been_ingested(sample_hash)
    finally:
        # cleanup any created files
        try:
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
        except Exception:
            pass
