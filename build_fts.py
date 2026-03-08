#!/usr/bin/env python3
"""
Build an FTS5 full-text search index over ALL components in cache.sqlite3.

This creates semantic/fts.sqlite3 with an FTS5 virtual table that enables
keyword matching to complement the FAISS semantic search.

Usage:
    python3 build_fts.py            # build (resumable)
    python3 build_fts.py --reset    # rebuild from scratch
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cache.sqlite3"
FTS_PATH = BASE_DIR / "semantic" / "fts.sqlite3"

DB_BATCH = 50_000


def build_search_text(mfr, package, description, extra_raw):
    """Build searchable text for FTS5 from all available fields."""
    tokens = []

    extra = None
    if extra_raw:
        try:
            extra = json.loads(extra_raw)
        except (json.JSONDecodeError, TypeError):
            pass

    if extra and isinstance(extra, dict):
        mpn = extra.get("mpn", "")
        if mpn:
            tokens.append(mpn)

        manufacturer = extra.get("manufacturer", {})
        if isinstance(manufacturer, dict):
            mfr_name = manufacturer.get("name", "")
            if mfr_name:
                tokens.append(mfr_name)

        cat = extra.get("category", {})
        if isinstance(cat, dict):
            for key in ("name1", "name2"):
                val = cat.get(key, "")
                if val:
                    tokens.append(val)

        attrs = extra.get("attributes", {})
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if v and v != "-":
                    tokens.append(f"{k} {v}")

        edesc = extra.get("description", "")
        if edesc:
            tokens.append(edesc)

        title = extra.get("title", "")
        if title:
            tokens.append(title)

    if mfr:
        tokens.append(mfr)
    if package:
        tokens.append(package)
    if description:
        tokens.append(description)

    return " ".join(tokens)


def get_min_price(extra_raw):
    """Extract lowest unit price from extra JSON. Returns None if unavailable."""
    if not extra_raw:
        return None
    try:
        extra = json.loads(extra_raw)
        prices = extra.get("prices", [])
        if prices:
            return min(p["price"] for p in prices if "price" in p)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Build FTS5 index for component search")
    parser.add_argument("--reset", action="store_true", help="Rebuild from scratch")
    args = parser.parse_args()

    FTS_PATH.parent.mkdir(exist_ok=True)

    if args.reset and FTS_PATH.exists():
        FTS_PATH.unlink()
        print(f"[Reset] Removed {FTS_PATH}")

    fts_conn = sqlite3.connect(str(FTS_PATH))
    fts_conn.execute("PRAGMA journal_mode=WAL")
    fts_conn.execute("PRAGMA synchronous=NORMAL")

    fts_conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS components_fts USING fts5(
            lcsc UNINDEXED,
            mfr,
            search_text,
            min_price UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        )
    """)

    fts_conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    row = fts_conn.execute("SELECT value FROM meta WHERE key='last_lcsc'").fetchone()
    last_lcsc = int(row[0]) if row else -1
    row = fts_conn.execute("SELECT value FROM meta WHERE key='total'").fetchone()
    total = int(row[0]) if row else 0

    src_conn = sqlite3.connect(str(DB_PATH))
    total_components = src_conn.execute("SELECT COUNT(*) FROM components").fetchone()[0]
    print(f"[FTS] Total components: {total_components:,}  |  Already indexed: {total:,}  |  Remaining: ~{total_components - total:,}")

    t_start = time.time()
    batch_num = 0

    while True:
        rows = src_conn.execute(
            "SELECT lcsc, mfr, package, description, extra FROM components WHERE lcsc > ? ORDER BY lcsc LIMIT ?",
            (last_lcsc, DB_BATCH),
        ).fetchall()

        if not rows:
            break

        fts_rows = []
        for lcsc, mfr, package, description, extra_raw in rows:
            text = build_search_text(mfr, package, description, extra_raw)
            price = get_min_price(extra_raw)
            fts_rows.append((lcsc, mfr, text, price))

        fts_conn.executemany(
            "INSERT INTO components_fts (lcsc, mfr, search_text, min_price) VALUES (?, ?, ?, ?)",
            fts_rows,
        )

        last_lcsc = rows[-1][0]
        total += len(rows)
        batch_num += 1

        fts_conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('last_lcsc', ?)", (str(last_lcsc),))
        fts_conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('total', ?)", (str(total),))
        fts_conn.commit()

        elapsed = time.time() - t_start
        rate = total / elapsed if elapsed > 0 else 0
        eta = (total_components - total) / rate if rate > 0 else 0
        print(f"  batch {batch_num:>4}  |  total {total:>9,}  |  {rate:,.0f} rows/s  |  ETA {eta:.0f}s  |  last_lcsc={last_lcsc}")

    src_conn.close()
    fts_conn.close()

    elapsed = time.time() - t_start
    print(f"\n[FTS] Done. {total:,} rows in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
