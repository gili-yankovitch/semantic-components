#!/usr/bin/env python3
"""
Index electronic components from cache.sqlite3 into a FAISS semantic vector DB.

Two-phase approach:
  Phase 1 (train): Sample components, embed them, train IVF centroids.
  Phase 2 (populate): Iterate all components in LCSC-ID order, embed and add
                       to the trained index. Resumable via checkpoint.

Usage:
    python3 index_components.py                  # full run (train if needed, then populate)
    python3 index_components.py --train-only     # only run training phase
    python3 index_components.py --reset          # wipe checkpoint and index, start fresh
"""

import argparse
import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cache.sqlite3"
SEMANTIC_DIR = BASE_DIR / "semantic"
INDEX_PATH = SEMANTIC_DIR / "index.faiss"
TRAINED_PATH = SEMANTIC_DIR / "index.trained"
CHECKPOINT_PATH = SEMANTIC_DIR / "checkpoint.json"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
NLIST = 4096          # IVF clusters
NPROBE = 64           # clusters to search (stored in index)
TRAIN_SAMPLE = 200_000
EMBED_BATCH = 512     # sentences per model.encode() call
DB_BATCH = 10_000     # rows fetched per SQL query
SAVE_EVERY = 50_000   # persist index every N vectors added

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[!] Shutdown requested – will save after current batch …")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"last_lcsc": -1, "total_indexed": 0, "phase": "train"}


def save_checkpoint(ckpt):
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    tmp.rename(CHECKPOINT_PATH)


def build_text(row):
    """Build a rich text representation of a component for embedding.

    Returns a text string for ALL components.  Components with rich extra
    JSON get a detailed representation; components with only the main-table
    columns get a minimal but still indexable string built from mfr + package
    + description.
    """
    lcsc, mfr, package, description, extra_raw = row

    if extra_raw:
        try:
            extra = json.loads(extra_raw)
        except (json.JSONDecodeError, TypeError):
            extra = None
    else:
        extra = None

    if not extra or not isinstance(extra, dict) or not extra.get("mpn"):
        parts = []
        if mfr:
            parts.append(mfr)
        if package:
            parts.append(f"Package: {package}")
        if description:
            parts.append(description)
        return " ".join(parts) if parts else mfr or f"C{lcsc}"

    parts = []
    manufacturer_name = extra.get("manufacturer", {}).get("name", mfr)
    mpn = extra.get("mpn", mfr)
    parts.append(f"{manufacturer_name} {mpn}")

    cat = extra.get("category", {})
    cat_parts = [cat.get("name1", ""), cat.get("name2", "")]
    cat_str = " - ".join(p for p in cat_parts if p)
    if cat_str:
        parts.append(cat_str)

    pkg = extra.get("package", package)
    if pkg:
        parts.append(f"Package: {pkg}")

    attrs = extra.get("attributes", {})
    if attrs:
        attr_strs = [f"{k}: {v}" for k, v in attrs.items() if v and v != "-"]
        if attr_strs:
            parts.append(", ".join(attr_strs))

    desc = extra.get("description", description)
    if desc:
        parts.append(desc)

    prices = extra.get("prices", [])
    if prices:
        try:
            lowest = min(p["price"] for p in prices if "price" in p)
            parts.append(f"Price: ${lowest:.4f}")
        except (ValueError, TypeError):
            pass

    return "\n".join(parts)


SELECT_COLS = "lcsc, mfr, package, description, extra"


def get_gpu_resource():
    """Return a FAISS GPU resource, or None if unavailable."""
    try:
        ngpu = faiss.get_num_gpus()
        if ngpu > 0:
            res = faiss.StandardGpuResources()
            return res
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Phase 1: Train
# ---------------------------------------------------------------------------

def phase_train(model):
    print(f"[Train] Sampling {TRAIN_SAMPLE:,} components for IVF training …")
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute(
        f"SELECT {SELECT_COLS} FROM components ORDER BY RANDOM() LIMIT ?",
        (TRAIN_SAMPLE,),
    )
    rows = cur.fetchall()
    conn.close()
    print(f"[Train] Fetched {len(rows):,} rows")

    texts = [build_text(r) for r in rows]
    print(f"[Train] Embedding {len(texts):,} texts …")
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)
    print(f"[Train] Embeddings shape: {embeddings.shape}")

    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, EMBEDDING_DIM, NLIST, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )

    gpu_res = get_gpu_resource()
    if gpu_res is not None:
        print("[Train] Training on GPU …")
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        gpu_index.train(embeddings)
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        print("[Train] Training on CPU …")
        index.train(embeddings)

    index.nprobe = NPROBE

    faiss.write_index(index, str(TRAINED_PATH))
    print(f"[Train] Trained index saved → {TRAINED_PATH}")
    return index


# ---------------------------------------------------------------------------
# Phase 2: Populate
# ---------------------------------------------------------------------------

def phase_populate(model, index, ckpt):
    last_lcsc = ckpt.get("last_lcsc", -1)
    total_indexed = ckpt.get("total_indexed", 0)
    unsaved = 0

    conn = sqlite3.connect(str(DB_PATH))

    total_rows = conn.execute("SELECT COUNT(*) FROM components").fetchone()[0]
    remaining = total_rows - total_indexed
    print(f"[Populate] Total components: {total_rows:,}  |  Already indexed: {total_indexed:,}  |  Remaining: ~{remaining:,}")

    wrapped = faiss.IndexIDMap2(index)
    if INDEX_PATH.exists() and total_indexed > 0:
        print(f"[Populate] Loading existing index with {total_indexed:,} vectors …")
        wrapped = faiss.read_index(str(INDEX_PATH))

    t_start = time.time()
    batch_num = 0

    while not _shutdown_requested:
        cur = conn.execute(
            f"SELECT {SELECT_COLS} FROM components WHERE lcsc > ? ORDER BY lcsc LIMIT ?",
            (last_lcsc, DB_BATCH),
        )
        rows = cur.fetchall()
        if not rows:
            break

        last_lcsc = int(rows[-1][0])

        ids = np.array([r[0] for r in rows], dtype=np.int64)
        texts = [build_text(r) for r in rows]

        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)

        wrapped.add_with_ids(embeddings, ids)

        total_indexed += len(rows)
        unsaved += len(rows)
        batch_num += 1

        elapsed = time.time() - t_start
        rate = total_indexed / elapsed if elapsed > 0 else 0
        eta = (total_rows - total_indexed) / rate if rate > 0 else 0
        print(
            f"  batch {batch_num:>5}  |  +{len(rows):>6}  |  total {total_indexed:>9,}  |  "
            f"{rate:,.0f} vec/s  |  ETA {eta/60:,.1f} min  |  last_lcsc={last_lcsc}"
        )

        if unsaved >= SAVE_EVERY:
            _save_state(wrapped, last_lcsc, total_indexed)
            unsaved = 0

    _save_state(wrapped, last_lcsc, total_indexed)
    conn.close()

    elapsed = time.time() - t_start
    print(f"\n[Populate] Done.  {total_indexed:,} vectors in {elapsed/60:.1f} min")
    if _shutdown_requested:
        print("[Populate] Interrupted – resume by running again.")
    return wrapped


def _save_state(index, last_lcsc, total_indexed):
    print(f"  ⟶  saving index ({total_indexed:,} vectors) …", end=" ", flush=True)
    t0 = time.time()
    faiss.write_index(index, str(INDEX_PATH))
    save_checkpoint({"last_lcsc": last_lcsc, "total_indexed": total_indexed, "phase": "populate"})
    print(f"done ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Index components into FAISS vector DB")
    parser.add_argument("--train-only", action="store_true", help="Only run the training phase")
    parser.add_argument("--reset", action="store_true", help="Delete checkpoint and index, start fresh")
    args = parser.parse_args()

    SEMANTIC_DIR.mkdir(exist_ok=True)

    if args.reset:
        for p in [INDEX_PATH, TRAINED_PATH, CHECKPOINT_PATH]:
            if p.exists():
                p.unlink()
                print(f"[Reset] Removed {p}")

    ckpt = load_checkpoint()

    device = "cuda" if _has_cuda() else "cpu"
    print(f"[Init] Loading embedding model '{MODEL_NAME}' on {device} …")
    model = SentenceTransformer(MODEL_NAME, device=device)

    need_train = not TRAINED_PATH.exists()
    if need_train or args.train_only:
        index = phase_train(model)
        ckpt["phase"] = "populate"
        save_checkpoint(ckpt)
        if args.train_only:
            return
    else:
        print(f"[Init] Trained index already exists → {TRAINED_PATH}")
        index = faiss.read_index(str(TRAINED_PATH))

    phase_populate(model, index, ckpt)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
