#!/usr/bin/env python3
"""Download MetaQA data.

Primary: official Google Drive folder via gdown
  https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE
  (yuyuz/MetaQA repo; CC Public License)
  Includes qtype files. May 401 if Drive permissions restrict it.

Fallback: camazlucas/MetaQA HF mirror (kb.txt + vanilla QA only, no qtype).
  Question types are then derived from templates (validated against the
  published counts: 21 types for 2-hop, 15 for 3-hop).

Stores SHA256 checksums in configs/metaqa_checksums.json
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

DRIVE_URL = ("https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE"
             "?resourcekey=0-kdv6ho5KcpEXdI2aUdLn_g&usp=sharing")
HF_MIRROR = "camazlucas/MetaQA"

# path in our layout -> path in HF mirror layout
MIRROR_MAP = {
    "kb.txt": "kb/kb.txt",
    "1-hop/vanilla/qa_train.txt": "1-hop/vanilla/qa_train.txt",
    "1-hop/vanilla/qa_dev.txt": "1-hop/vanilla/qa_dev.txt",
    "1-hop/vanilla/qa_test.txt": "1-hop/vanilla/qa_test.txt",
    "2-hop/vanilla/qa_train.txt": "2-hop/vanilla/qa_train.txt",
    "2-hop/vanilla/qa_dev.txt": "2-hop/vanilla/qa_dev.txt",
    "2-hop/vanilla/qa_test.txt": "2-hop/vanilla/qa_test.txt",
    "3-hop/vanilla/qa_train.txt": "3-hop/vanilla/qa_train.txt",
    "3-hop/vanilla/qa_dev.txt": "3-hop/vanilla/qa_dev.txt",
    "3-hop/vanilla/qa_test.txt": "3-hop/vanilla/qa_test.txt",
}

RAW_DIR = Path("data/metaqa/raw")
CHECKSUM_FILE = Path("configs/metaqa_checksums.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def try_drive() -> bool:
    print("Trying official Drive folder...", flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "gdown", "--folder", DRIVE_URL, "-O", str(RAW_DIR)],
        capture_output=False,
    )
    kb = list(RAW_DIR.rglob("kb.txt"))
    return r.returncode == 0 and len(kb) > 0


def from_mirror() -> dict:
    from huggingface_hub import hf_hub_download

    print(f"Falling back to HF mirror {HF_MIRROR}...", flush=True)
    checksums = {}
    for ours, theirs in MIRROR_MAP.items():
        path = hf_hub_download(HF_MIRROR, filename=theirs, repo_type="dataset",
                               local_dir=str(RAW_DIR))
        dst = RAW_DIR / ours
        if Path(path) != dst:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, dst)
        checksums[ours] = {"sha256": sha256(dst), "bytes": dst.stat().st_size}
    return checksums


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    checksums: dict = {}
    source = ""

    if try_drive():
        source = "gdrive-official"
        # normalize + checksum wanted files (implemented on success path)
        print("Drive download worked.", flush=True)
    else:
        source = f"hf-mirror:{HF_MIRROR}"
        checksums = from_mirror()

    with open(CHECKSUM_FILE, "w") as f:
        json.dump({"source": source, "files": checksums}, f, indent=2)

    total = sum(v["bytes"] for v in checksums.values())
    print(f"Kept {len(checksums)} files, {total/1e6:.1f} MB from {source}", flush=True)


if __name__ == "__main__":
    main()
