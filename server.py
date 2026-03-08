#!/usr/bin/env python3
"""
Persistent HTTP search server for the semantic electronic components database.

Keeps the FAISS index, FTS5 database, embedding model, and component DB
loaded in memory. Exposes a /search endpoint for hybrid queries combining
semantic vector search, FTS5 keyword search, and direct part-number matching.

Start:
    python3 server.py                     # default port 8811
    python3 server.py --port 9000 --cpu   # custom port, force CPU
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cache.sqlite3"
FTS_PATH = BASE_DIR / "semantic" / "fts.sqlite3"
INDEX_PATH = BASE_DIR / "semantic" / "index.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60             # reciprocal-rank-fusion constant
FAISS_NPROBE = 64
DEFAULT_TOP = 10
FAISS_OVERSAMPLE = 10  # fetch N*top from FAISS for better RRF coverage
FTS_OVERSAMPLE = 5

logger = logging.getLogger("parts-server")

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
state: dict = {}


def _detect_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def startup(use_gpu: bool = True):
    device = _detect_device() if use_gpu else "cpu"
    logger.info("Loading embedding model '%s' on %s …", MODEL_NAME, device)
    state["model"] = SentenceTransformer(MODEL_NAME, device=device)

    logger.info("Loading FAISS index from %s …", INDEX_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    if use_gpu and device == "cuda":
        try:
            ngpu = faiss.get_num_gpus()
            if ngpu > 0:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index transferred to GPU")
        except Exception as exc:
            logger.warning("GPU transfer failed (%s), using CPU index", exc)

    state["index"] = index

    logger.info("Opening FTS database %s …", FTS_PATH)
    state["fts_conn"] = sqlite3.connect(str(FTS_PATH), check_same_thread=False)

    logger.info("Opening component database %s …", DB_PATH)
    state["db_conn"] = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    state["db_conn"].execute("PRAGMA mmap_size=2147483648")

    logger.info("Server ready.")


def shutdown():
    for key in ("fts_conn", "db_conn"):
        conn = state.get(key)
        if conn:
            conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup(use_gpu=state.get("use_gpu", True))
    yield
    shutdown()


app = FastAPI(title="Parts Semantic Search", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Component(BaseModel):
    lcsc: int
    mfr: str
    mpn: Optional[str] = None
    manufacturer: Optional[str] = None
    category: Optional[str] = None
    package: Optional[str] = None
    description: Optional[str] = None
    attributes: Optional[dict] = None
    price: Optional[float] = None
    stock: int = 0
    score: float = 0.0
    match_sources: list[str] = []


class SearchResponse(BaseModel):
    query: str
    count: int
    elapsed_ms: float
    results: list[Component]

# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------

_FTS_STOP = frozenset("a an the to for with from by in on of and or not is are was be".split())


def _tokenize_like_fts5(text: str) -> list[str]:
    """Split text into tokens the same way FTS5 unicode61 does -- on any
    non-alphanumeric character.  Drops pure-digit tokens of 1-2 chars
    (like '3' from '3.3v') since they are too broad for prefix search."""
    out = []
    for t in re.split(r'[^a-zA-Z0-9]+', text):
        if not t:
            continue
        if len(t) <= 2 and t.isdigit():
            continue
        out.append(t)
    return out


def _make_fts_query(raw_query: str) -> str:
    """Convert a user query into an FTS5 query with prefix matching.

    - Input is split into sub-tokens matching the FTS5 unicode61 tokenizer.
    - Stop words are removed.
    - Multi-token queries use AND (all terms must match).
    - Each token gets a trailing * for prefix matching so that partial part
      numbers work (e.g. 'ch32v003' matches 'CH32V003F4P6').
    - Quoted phrases are preserved.
    """
    tokens: list[str] = []
    in_quote = False
    current: list[str] = []

    for ch in raw_query:
        if ch == '"':
            if in_quote:
                tokens.append('"' + "".join(current) + '"')
                current = []
                in_quote = False
            else:
                if current:
                    for sub in _tokenize_like_fts5("".join(current)):
                        if sub.lower() not in _FTS_STOP:
                            tokens.append(sub + "*")
                    current = []
                in_quote = True
        elif ch in (' ', '\t'):
            if in_quote:
                current.append(ch)
            else:
                if current:
                    for sub in _tokenize_like_fts5("".join(current)):
                        if sub.lower() not in _FTS_STOP:
                            tokens.append(sub + "*")
                    current = []
        else:
            current.append(ch)

    if current:
        if in_quote:
            tokens.append('"' + "".join(current) + '"')
        else:
            for sub in _tokenize_like_fts5("".join(current)):
                if sub.lower() not in _FTS_STOP:
                    tokens.append(sub + "*")

    if not tokens:
        return raw_query

    if len(tokens) == 1:
        return tokens[0]

    return " AND ".join(tokens)


def search_faiss(query: str, top_k: int) -> list[tuple[int, float]]:
    """Semantic search via FAISS. Returns [(lcsc_id, score), ...]."""
    model = state["model"]
    index = state["index"]

    embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, ids = index.search(embedding, top_k)

    results = []
    for dist, lid in zip(distances[0], ids[0]):
        if lid >= 0:
            results.append((int(lid), float(dist)))
    return results


def search_fts(query: str, top_k: int) -> list[tuple[int, float, float]]:
    """FTS5 keyword search. Returns [(lcsc_id, fts_rank, min_price), ...]."""
    conn = state["fts_conn"]
    fts_query = _make_fts_query(query)

    multi_token = " AND " in fts_query or " OR " in fts_query
    if multi_token:
        fts_query = f"search_text: ({fts_query})"

    try:
        rows = conn.execute(
            "SELECT lcsc, rank, min_price FROM components_fts WHERE components_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    return [(int(r[0]), float(r[1]), r[2]) for r in rows]


_STOP_WORDS = frozenset(
    "a an the to for with from by in on of and or not is are was be "
    "cheap expensive good best small large high low micro mini".split()
)


def _looks_like_part_number(query: str) -> bool:
    """Heuristic: a query is a part-number lookup only if it's 1-2 tokens
    with at least one containing both letters and digits, and none are
    common English words."""
    tokens = query.strip().lower().split()
    if len(tokens) > 2:
        return False
    if any(t in _STOP_WORDS for t in tokens):
        return False
    has_alnum = lambda t: bool(re.search(r'[a-zA-Z]', t) and re.search(r'\d', t))
    return any(has_alnum(t) for t in tokens)


MFR_SEARCH_TIMEOUT_MS = 2000


def search_mfr(query: str, top_k: int) -> list[int]:
    """Direct part-number search on the mfr column. Returns [lcsc_id, ...].

    Only runs for queries that look like part numbers to avoid expensive
    full-table LIKE scans on natural language queries.  A progress-handler
    timeout aborts scans that exceed MFR_SEARCH_TIMEOUT_MS.
    """
    clean = query.strip()
    if len(clean) < 2 or not _looks_like_part_number(clean):
        return []

    conn = state["db_conn"]

    deadline = time.monotonic() + MFR_SEARCH_TIMEOUT_MS / 1000.0

    def _progress():
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_progress, 10000)
    try:
        tokens = clean.split()
        if len(tokens) == 1:
            rows = conn.execute(
                "SELECT lcsc FROM components WHERE mfr LIKE ? LIMIT ?",
                (f"%{clean}%", top_k),
            ).fetchall()
        else:
            conditions = " AND ".join(["mfr LIKE ?"] * len(tokens))
            params = [f"%{t}%" for t in tokens] + [top_k]
            rows = conn.execute(
                f"SELECT lcsc FROM components WHERE {conditions} LIMIT ?",
                params,
            ).fetchall()
    except sqlite3.OperationalError:
        logger.warning("MFR search timed out for query: %s", clean)
        return []
    finally:
        conn.set_progress_handler(None, 0)

    return [int(r[0]) for r in rows]


STOCK_BOOST_WEIGHT = 0.15


def hybrid_search(query: str, top_k: int, sort_by_price: bool = False) -> list[Component]:
    """Merge semantic + FTS5 + mfr results via Reciprocal Rank Fusion.

    After RRF merging, a small stock-based boost is applied so that
    popular, in-stock components surface above obscure zero-stock parts
    with similar relevance scores.
    """

    faiss_k = top_k * FAISS_OVERSAMPLE
    fts_k = top_k * FTS_OVERSAMPLE

    faiss_results = search_faiss(query, faiss_k)
    fts_results = search_fts(query, fts_k)
    mfr_results = search_mfr(query, top_k)

    rrf_scores: dict[int, float] = {}
    sources: dict[int, list[str]] = {}

    for rank, (lid, _score) in enumerate(faiss_results, 1):
        rrf_scores[lid] = rrf_scores.get(lid, 0.0) + 1.0 / (RRF_K + rank)
        sources.setdefault(lid, []).append("semantic")

    for rank, (lid, _fts_rank, _price) in enumerate(fts_results, 1):
        rrf_scores[lid] = rrf_scores.get(lid, 0.0) + 1.0 / (RRF_K + rank)
        sources.setdefault(lid, []).append("keyword")

    for rank, lid in enumerate(mfr_results, 1):
        boost = 2.0 / (RRF_K + rank)
        rrf_scores[lid] = rrf_scores.get(lid, 0.0) + boost
        sources.setdefault(lid, []).append("partnumber")

    if not rrf_scores:
        return []

    scored = sorted(rrf_scores.items(), key=lambda x: -x[1])
    candidate_ids = [lid for lid, _ in scored[:top_k * 3]]

    components = _fetch_components(candidate_ids)

    if components:
        max_rrf = max(rrf_scores.values())
        for comp in components:
            base = rrf_scores.get(comp.lcsc, 0.0)
            stock_factor = min(comp.stock / 100_000, 1.0) if comp.stock > 0 else 0.0
            comp.score = base + stock_factor * max_rrf * STOCK_BOOST_WEIGHT
            comp.match_sources = sources.get(comp.lcsc, [])

    if sort_by_price:
        components.sort(key=lambda c: (c.price if c.price is not None else float("inf"), -c.score))
    else:
        components.sort(key=lambda c: -c.score)

    return components[:top_k]


def _fetch_components(lcsc_ids: list[int]) -> list[Component]:
    """Look up full component info from cache.sqlite3."""
    if not lcsc_ids:
        return []

    conn = state["db_conn"]
    placeholders = ",".join("?" * len(lcsc_ids))
    rows = conn.execute(
        f"SELECT lcsc, mfr, package, description, stock, price, extra "
        f"FROM components WHERE lcsc IN ({placeholders})",
        lcsc_ids,
    ).fetchall()

    lookup = {r[0]: r for r in rows}
    components = []

    for lid in lcsc_ids:
        row = lookup.get(lid)
        if not row:
            continue

        lcsc, mfr, package, description, stock, price_raw, extra_raw = row

        extra = None
        if extra_raw:
            try:
                extra = json.loads(extra_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        comp = Component(lcsc=lcsc, mfr=mfr, package=package, stock=stock)

        if extra and isinstance(extra, dict):
            comp.mpn = extra.get("mpn", mfr)
            comp.manufacturer = extra.get("manufacturer", {}).get("name")

            cat = extra.get("category", {})
            cat_parts = [cat.get("name1", ""), cat.get("name2", "")]
            comp.category = " > ".join(p for p in cat_parts if p) or None

            comp.description = extra.get("description", description) or description

            attrs = extra.get("attributes", {})
            if attrs:
                comp.attributes = {k: v for k, v in attrs.items() if v and v != "-"}

            prices = extra.get("prices", [])
            if prices:
                try:
                    comp.price = min(p["price"] for p in prices if "price" in p)
                except (ValueError, TypeError):
                    pass
        else:
            comp.mpn = mfr
            comp.description = description or None

        components.append(comp)

    return components


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/search", response_model=SearchResponse)
def api_search(
    q: str = Query(..., description="Search query"),
    top: int = Query(DEFAULT_TOP, ge=1, le=200, description="Max results"),
    sort: str = Query("relevance", description="Sort order: 'relevance' or 'price'"),
):
    t0 = time.time()
    sort_by_price = sort.lower() == "price"
    results = hybrid_search(q, top, sort_by_price=sort_by_price)
    elapsed = (time.time() - t0) * 1000

    return SearchResponse(
        query=q,
        count=len(results),
        elapsed_ms=round(elapsed, 1),
        results=results,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "faiss_vectors": state["index"].ntotal if "index" in state else 0,
        "model": MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parts semantic search server")
    parser.add_argument("--port", type=int, default=8811, help="Listen port (default: 8811)")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    state["use_gpu"] = not args.cpu

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
