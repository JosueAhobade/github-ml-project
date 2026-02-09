#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter Hugging Face dataset in streaming mode to keep only issues from TARGET_REPOS,
and write results to a compressed JSONL (.jsonl.gz) file.

Usage:
  pip install -U datasets huggingface_hub tqdm

  python filter_top20_issues.py

Optional:
  export HF_HOME=/path/to/cache
  huggingface-cli login  (if needed)
"""

import os
import sys
import json
import gzip
from collections import Counter
from datetime import datetime

from datasets import load_dataset
from tqdm import tqdm


# -------------------------
# 1) CONFIG
# -------------------------

DATASET_NAME = "bigcode/the-stack-github-issues"
SPLIT = "train"

# Your 20 famous repos (full_name: owner/repo)
TARGET_REPOS = {
    "iluwatar/java-design-patterns", "airbnb/lottie-android", "nationalsecurityagency/ghidra",
    "apache/rocketmq", "material-components/material-components-android", "thealgorithms/java",
    "openapitools/openapi-generator", "thingsboard/thingsboard", "dolphinscheduler/dolphinscheduler",
    "alibaba/canal", "google/ios-sched", "spring-cloud/spring-cloud-gateway",
    "public-apis/public-apis", "donnemartin/system-design-primer", "python/cpython",
    "vinta/awesome-python", "ytdl-org/youtube-dl", "tensorflow/models"
}

# Output
OUT_PATH = "issues_top20.jsonl.gz"

# If you already know the repo field name, set it here (e.g. "repo", "repository", "full_name")
# If None, the script will auto-detect.
REPO_FIELD_OVERRIDE = None

# Write progress / flush frequency
FLUSH_EVERY = 2000  # write buffer flush interval (lines)


# -------------------------
# 2) HELPERS
# -------------------------

def detect_repo_field(sample: dict) -> str:
    """
    Try to detect which key contains the repository full name (owner/repo).
    Returns the best guess or raises ValueError.
    """
    candidates = ["repo", "repository", "full_name", "repo_name", "repository_full_name"]
    keys = set(sample.keys())

    for c in candidates:
        if c in keys and isinstance(sample.get(c), str) and "/" in sample.get(c):
            return c

    # Fallback: scan any string field containing "owner/repo"
    for k, v in sample.items():
        if isinstance(v, str) and "/" in v and len(v) < 200:
            # crude heuristic
            if v.count("/") == 1 and " " not in v:
                return k

    raise ValueError(
        f"Impossible de détecter la colonne repo. Clés disponibles: {sorted(list(keys))[:60]}..."
    )


def normalize_repo(repo_val: str) -> str:
    """
    Normalize repo string (strip, lower?).
    GitHub full_name is case-insensitive, but datasets typically store as given.
    We'll keep exact but strip spaces.
    """
    if repo_val is None:
        return ""
    return str(repo_val).strip()


# -------------------------
# 3) MAIN
# -------------------------

def main():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Loading dataset in streaming mode...")
    ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

    # Peek one sample to detect field names
    it = iter(ds)
    try:
        first = next(it)
    except StopIteration:
        print("Dataset vide ?", file=sys.stderr)
        sys.exit(1)

    # Detect repo field
    repo_field = REPO_FIELD_OVERRIDE or detect_repo_field(first)
    print(f"[INFO] Repo field detected: '{repo_field}'")

    # Put back first element by processing it manually then continuing with iterator
    counter = Counter()
    total_seen = 0
    total_kept = 0
    buffer_lines = []

    # Open output
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Writing to: {OUT_PATH}")
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:

        def process_row(row: dict):
            nonlocal total_seen, total_kept, buffer_lines
            total_seen += 1

            repo_val = normalize_repo(row.get(repo_field, ""))
            if repo_val in TARGET_REPOS:
                total_kept += 1
                counter[repo_val] += 1
                buffer_lines.append(json.dumps(row, ensure_ascii=False))

                if len(buffer_lines) >= FLUSH_EVERY:
                    f.write("\n".join(buffer_lines) + "\n")
                    buffer_lines.clear()

        # process first row
        process_row(first)

        # process remaining rows
        for row in tqdm(it, desc="Streaming & filtering"):
            process_row(row)

        # flush remaining
        if buffer_lines:
            f.write("\n".join(buffer_lines) + "\n")
            buffer_lines.clear()

    # Summary
    print("\n================ SUMMARY ================")
    print(f"Total rows scanned: {total_seen}")
    print(f"Total rows kept  : {total_kept}")
    print("Kept per repo:")
    for repo, n in counter.most_common():
        print(f"  - {repo}: {n}")
    print("========================================\n")

    print(f"[DONE] Output file: {OUT_PATH}")
    print("Next step: import this jsonl.gz into your DB, then run enrichment (repos/users/commits).")


if __name__ == "__main__":
    main()