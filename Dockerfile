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

# Pre-download the embedding model into the image
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY server.py .

EXPOSE 8811

CMD ["python3", "server.py", "--port", "8811"]
