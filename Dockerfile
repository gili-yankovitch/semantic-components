FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu126

RUN pip3 install --no-cache-dir \
    faiss-gpu-cu12 \
    sentence-transformers \
    fastapi \
    uvicorn

RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ----- Layer strategy -----
# Each COPY = one registry layer, pushed independently by digest.
# Order from least-frequently to most-frequently changed so that
# "docker push" after a code edit uploads only the last ~16 KB layer.
#
#   Layer  What                   Size    Changes when
#   ─────  ─────────────────────  ──────  ─────────────────────────
#   1      cache.sqlite3          ~12 GB  upstream jlcparts updates
#   2      semantic/index.faiss   ~1.6 GB re-run index_components.py
#   3      semantic/fts.sqlite3   ~2.5 GB re-run build_fts.py
#   4      server.py              ~16 KB  any code change

# Layer 1 — component database (changes only on data refresh)
COPY cache.sqlite3 .

# Layer 2 — FAISS vector index (changes on re-index)
COPY semantic/index.faiss ./semantic/

# Layer 3 — FTS5 keyword index (changes on re-index)
COPY semantic/fts.sqlite3 ./semantic/

# Layer 4 — application code (changes frequently)
COPY server.py .

EXPOSE 8811

CMD ["python3", "server.py", "--port", "8811"]
