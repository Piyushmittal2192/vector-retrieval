# syntax=docker/dockerfile:1
# Retrieval service image: CPU-only torch, pinned model baked in, non-root,
# works with a read-only root filesystem (only /tmp and /index are writable).
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS build
# CPU wheels by default. For a GPU encoder image pass the CUDA wheel index that
# matches your node's driver (see pytorch.org), e.g.
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX ...
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # hnswlib compiles with -march=native by default: fast on THIS cpu, but an
    # image built on your laptop can crash with "illegal instruction" on a
    # cloud node with an older CPU. Portable build:
    HNSWLIB_NO_NATIVE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements/ /tmp/req/
RUN pip install --upgrade pip \
 && if [ -f /tmp/req/serve.lock ]; then \
      echo "installing from serve.lock" \
      && pip install --extra-index-url "$TORCH_INDEX_URL" -r /tmp/req/serve.lock; \
    else \
      echo "no serve.lock yet: resolving ranges (run 'make lock' afterwards)" \
      && pip install --index-url "$TORCH_INDEX_URL" "torch>=2.3" \
      && pip install --extra-index-url "$TORCH_INDEX_URL" -r /tmp/req/serve.in; \
    fi \
 && pip freeze > /opt/venv/requirements.freeze

FROM ${PYTHON_IMAGE}
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /opt/venv /opt/venv
COPY models/ /models/
COPY retrieval/ /app/retrieval/
WORKDIR /app
ARG BUILD_ID=dev
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app HOME=/tmp \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf \
    ENCODER_BACKEND=st MODEL_PATH=/models/all-MiniLM-L6-v2 \
    MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
    TORCH_THREADS=2 OMP_NUM_THREADS=2 \
    BUILD_ID=${BUILD_ID}
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "retrieval.server:app", "--host", "0.0.0.0", "--port", "8000", \
     "--no-access-log", "--timeout-graceful-shutdown", "20", "--timeout-keep-alive", "5"]
