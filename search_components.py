#!/usr/bin/env python3
"""
Lightweight CLI client for the parts semantic search server.

Usage:
    python3 search_components.py "LDO to 3.3v"
    python3 search_components.py "cheap riscv mcu" --top 20 --sort price
    python3 search_components.py --interactive
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

SERVER_URL = "http://localhost:8811"
DEFAULT_TOP = 10


def search(query: str, top: int = DEFAULT_TOP, sort: str = "relevance", server: str = SERVER_URL) -> dict:
    params = urllib.parse.urlencode({"q": query, "top": top, "sort": sort})
    url = f"{server}/search?{params}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"Error: cannot reach server at {server} ({e})", file=sys.stderr)
        print("Start the server first:  python3 server.py", file=sys.stderr)
        sys.exit(1)


def format_results(data: dict):
    query = data["query"]
    count = data["count"]
    elapsed = data["elapsed_ms"]

    print(f"\n  Results for: \"{query}\"  ({count} results, {elapsed:.0f}ms)\n")

    for rank, comp in enumerate(data["results"], 1):
        lcsc = comp["lcsc"]
        mpn = comp.get("mpn") or comp.get("mfr", "")
        manufacturer = comp.get("manufacturer") or ""
        category = comp.get("category") or ""
        package = comp.get("package") or ""
        description = comp.get("description") or ""
        price = comp.get("price")
        stock = comp.get("stock", 0)
        score = comp.get("score", 0)
        sources = comp.get("match_sources", [])
        attrs = comp.get("attributes") or {}

        price_str = f"${price:.4f}" if price is not None else "N/A"
        source_str = "+".join(sources)
        attr_strs = [f"{k}: {v}" for k, v in attrs.items()]

        print(f"{'─' * 78}")
        print(f"  #{rank:<3}  C{lcsc}  |  {manufacturer} {mpn}  |  score: {score:.4f}  [{source_str}]")
        if category:
            print(f"        Category: {category}")
        print(f"        Package:  {package}    Stock: {stock:,}    Price: {price_str}")
        if attr_strs:
            print(f"        Attrs:    {', '.join(attr_strs[:6])}")
            if len(attr_strs) > 6:
                print(f"                  {', '.join(attr_strs[6:])}")
        if description:
            desc = description[:120] + "…" if len(description) > 120 else description
            print(f"        {desc}")

    print(f"{'─' * 78}")


def main():
    parser = argparse.ArgumentParser(description="Search electronic components")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="Number of results (default: 10)")
    parser.add_argument("--sort", choices=["relevance", "price"], default="relevance", help="Sort order")
    parser.add_argument("--server", default=SERVER_URL, help=f"Server URL (default: {SERVER_URL})")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.interactive or args.query is None:
        print("\nSemantic Component Search (type 'quit' to exit)")
        print(f"Server: {args.server}\n")
        while True:
            try:
                query = input("search> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break

            data = search(query, args.top, args.sort, args.server)
            if args.json:
                print(json.dumps(data, indent=2))
            else:
                format_results(data)
            print()
    else:
        data = search(args.query, args.top, args.sort, args.server)
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            format_results(data)


if __name__ == "__main__":
    main()
