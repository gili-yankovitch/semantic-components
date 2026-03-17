# Parts Semantic Search

GPU-accelerated semantic search over the ~7 million electronic components in
the [JLCPCB SMT assembly catalogue](https://jlcpcb.com/parts). Ask natural
language questions like *"cheap RISC-V microcontroller"* or *"LDO to 3.3v"*
and get ranked results combining three search strategies:

- **Semantic** -- FAISS vector search using sentence-transformer embeddings on GPU
- **Keyword** -- SQLite FTS5 full-text search with prefix matching
- **Part number** -- Direct manufacturer part number lookup (e.g. `STM32F4`, `CH32V003`)

Results are merged via Reciprocal Rank Fusion (RRF) with an in-stock boost so
that popular, available parts surface first.

## Architecture

```
┌──────────────┐
│  cache.sqlite3  │  ~12 GB, 7M components (from jlcparts)
└──────┬───────┘
       │
       ├── index_components.py ──► semantic/index.faiss   (~1.6 GB, FAISS IVF-SQ8)
       ├── build_fts.py        ──► semantic/fts.sqlite3   (~2.5 GB, FTS5 keyword index)
       │
       ▼
┌──────────────┐     GET /search?q=...     ┌─────────────────┐
│   server.py     │◄────────────────────────│  search_components.py  │
│   (FastAPI)     │        :8811            │  (CLI client)          │
│   port 8811     │◄───────┐               └─────────────────┘
└──────────────┘   │
                   │  /api/* proxy
              ┌────┴───────────┐
              │   web UI (nginx)  │
              │   port 3080       │
              └────────────────┘
```

## Prerequisites - indexing the DB should be preformed only on cuda enabled system

- **NVIDIA GPU** with CUDA support (tested on RTX 4070)
- **NVIDIA drivers** + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for Docker)
- Python 3.10+ (for native install)

### Python dependencies (native install)

```
torch                  (with CUDA, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu126)
faiss-gpu-cu12
sentence-transformers
fastapi
uvicorn
numpy
```

## Getting the Component Database

The `cache.sqlite3` database comes from the
[jlcparts](https://github.com/yaqwsx/jlcparts) project, which scrapes the
full JLCPCB/LCSC component catalogue.

### Option A: Download pre-built cache (recommended)

Download the split-zip archive published by the jlcparts GitHub Pages build:

```bash
wget https://yaqwsx.github.io/jlcparts/data/cache.zip
wget https://yaqwsx.github.io/jlcparts/data/cache.z{01..18}
7z x cache.zip
```

This produces `cache.sqlite3` (~12 GB). Place it in the project root.

### Option B: Build from source

```bash
git clone https://github.com/yaqwsx/jlcparts
cd jlcparts
pip install -e .
jlcparts getLibrary cache.sqlite3
```

This fetches component data directly from LCSC's API and builds the database
from scratch. It takes several hours and many API calls.

## Building the Search Indexes

Two indexes must be built from `cache.sqlite3` before the server can start.
Both scripts are resumable -- if interrupted, re-run and they pick up where
they left off.

### 1. FAISS semantic vector index

```bash
python3 index_components.py
```

This runs in two phases:

1. **Train** -- Samples 200K components, embeds them on GPU, trains 4096 IVF
   centroids. Saves `semantic/index.trained`.
2. **Populate** -- Iterates all 7M components in LCSC-ID order, embeds in
   batches of 512, and adds to the index. Saves progress every 50K vectors.
   Resumable via `semantic/checkpoint.json`.

Estimated time: **30--60 minutes** on an RTX 4070. Output: `semantic/index.faiss` (~1.6 GB).

Useful flags:

```bash
python3 index_components.py --train-only   # only train centroids
python3 index_components.py --reset        # wipe everything and start fresh
```

Press Ctrl+C to interrupt gracefully -- the current batch is saved before exit.

### 2. FTS5 full-text keyword index

```bash
python3 build_fts.py
```

Iterates all components and builds an FTS5 virtual table with manufacturer
part numbers, category names, attributes, and descriptions. Resumable.

Estimated time: **2--5 minutes**. Output: `semantic/fts.sqlite3` (~2.5 GB).

```bash
python3 build_fts.py --reset   # rebuild from scratch
```

## Running the Server

### Native

```bash
python3 server.py                     # default: 0.0.0.0:8811, GPU mode
python3 server.py --port 9000         # custom port
python3 server.py --cpu               # force CPU mode (slower)
```

The server loads the embedding model, FAISS index, and both SQLite databases
into memory at startup (~10 seconds), then serves requests on the configured
port.

### Docker Compose (GPU)

```bash
docker compose build
docker compose up -d
```
### Docker Compose (CPU/MAC)

```bash
docker compose -f docker-compose.CPU_MAC.yml build
docker compose -f docker-compose.CPU_MAC.yml up
```



This starts two containers:

- **search** (port 8811) -- FastAPI backend with GPU/CUDA
- **web** (port 3080) -- Web UI served by nginx, proxies `/api/*` to the backend

Open **http://localhost:3080** for the web UI.

The search backend uses `nvidia/cuda:12.6.3-runtime-ubuntu22.04` and requests
GPU access via the NVIDIA Container Toolkit. The data files (`cache.sqlite3`,
`index.faiss`, `fts.sqlite3`) are baked into the image as separate Docker
layers so that code-only changes result in tiny pushes to a container
registry. The web UI is a lightweight `nginx:alpine` container.

## Web UI

The web UI is a single-page app at `web/index.html` served by nginx on port
3080. It provides a search bar with example queries, sort controls, and
displays results as cards with part number, manufacturer, category,
attributes, price, and stock info. Each result links to the LCSC product page.

The nginx config in `web/nginx.conf` reverse-proxies `/api/*` requests to the
`search` backend container so there are no CORS issues in the browser.

## API

### `GET /search`

| Parameter | Type   | Default     | Description                         |
|-----------|--------|-------------|-------------------------------------|
| `q`       | string | (required)  | Search query                        |
| `top`     | int    | 10          | Max results (1--200)                |
| `sort`    | string | `relevance` | Sort order: `relevance` or `price`  |

Example:

```bash
curl 'http://localhost:8811/search?q=LDO+to+3.3v&top=5'
```

Response:

```json
{
  "query": "LDO to 3.3v",
  "count": 5,
  "elapsed_ms": 126.0,
  "results": [
    {
      "lcsc": 123456,
      "mfr": "XC6206P332MR",
      "mpn": "XC6206P332MR",
      "manufacturer": "Torex Semicon",
      "category": "Power Management > LDO",
      "package": "SOT-23-3",
      "description": "3.3V 200mA Low Dropout Regulator",
      "attributes": {"Output Voltage": "3.3V", "Output Current": "200mA"},
      "price": 0.0198,
      "stock": 584200,
      "score": 0.0327,
      "match_sources": ["semantic", "keyword"]
    }
  ]
}
```

### `GET /health`

Returns server status and vector count:

```json
{"status": "ok", "faiss_vectors": 4091139, "model": "all-MiniLM-L6-v2"}
```

## CLI Client

A lightweight CLI client that talks to the running server:

```bash
python3 search_components.py "LDO to 3.3v"
python3 search_components.py "cheap riscv mcu" --top 20 --sort price
python3 search_components.py --interactive
python3 search_components.py "STM32F4" --json
```

The client has no GPU dependencies -- it just makes HTTP requests.

## File Structure

```
.
├── server.py               # FastAPI search server (hybrid semantic + FTS + part number)
├── index_components.py     # Build FAISS vector index from cache.sqlite3 (GPU)
├── build_fts.py            # Build FTS5 keyword index from cache.sqlite3
├── search_components.py    # CLI client for the search server
├── cache.sqlite3           # Component database (~12 GB, not in repo)
├── semantic/
│   ├── index.faiss         # FAISS IVF-SQ8 vector index (~1.6 GB)
│   ├── index.trained       # Trained empty index (for recovery)
│   ├── fts.sqlite3         # FTS5 keyword index (~2.5 GB)
│   └── checkpoint.json     # Indexing progress tracker
├── web/
│   ├── index.html          # Web UI (single-page app)
│   ├── nginx.conf          # nginx config with /api reverse proxy
│   └── Dockerfile          # nginx:alpine image for the web UI
├── Dockerfile              # CUDA image for the search backend
├── docker-compose.yml      # Brings up search + web containers
├── .dockerignore
└── jlcparts/               # Upstream jlcparts repo (builds cache.sqlite3)
```
