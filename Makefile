# Production vector retrieval lab. `make help` lists targets.
# Every target is a thin wrapper; read the recipe to see the real command.
SHELL := /bin/bash
.DEFAULT_GOAL := help

PY       ?= .venv/bin/python
TIER     ?= xs
# ENCODER: st = sentence-transformers (real model); hash = lexical stand-in, no model needed
ENCODER  ?= st
QUERIES  ?= 2000
CHANGES  ?= 5000
V        ?= v1
BASELINE ?=
TAG      ?= 1.0
IMAGE    ?= retrieval-api:$(TAG)
LOADGEN_IMAGE ?= retrieval-loadgen:1.0
COLOR    ?= blue
REPLICAS ?= 3
NAME     ?= run-$(shell date +%H%M%S)
RATE     ?= 100
DURATION ?= 120
RATES    ?= 25,50,100,150,200,300,400

ifeq ($(TIER),xs)
PRODUCTS ?= 20000
PAD ?= 0
else ifeq ($(TIER),s)
PRODUCTS ?= 100000
PAD ?= 0
else ifeq ($(TIER),m)
PRODUCTS ?= 300000
PAD ?= 0
else ifeq ($(TIER),l)
PRODUCTS ?= 300000
PAD ?= 1700000
BUILD_ARGS ?= --ef-construction 100
endif
export IMAGE LOADGEN_IMAGE

help: ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

# ---------------------------------------------------------------- laptop
venv: ## create .venv with CPU-only torch and all Python deps
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3"
	$(PY) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/dev.in

test: ## unit + integration tests (hash encoder, no model needed)
	$(PY) -m pytest -q

sizing: ## suggest TIER and REPLICAS for this machine
	@ops/sizing.sh

data: ## generate catalog, labeled queries, empty change log (TIER=xs|s|m|l)
	$(PY) -m data.generate_catalog --n $(PRODUCTS)
	$(PY) -m data.generate_queries --n $(QUERIES)
	: > data/changes.jsonl
	mkdir -p artifacts/queries && cp data/queries.jsonl artifacts/queries/queries.jsonl && chmod -R a+rX artifacts

changes: ## append CHANGES product events to the change log
	$(PY) -m data.generate_changes --n $(CHANGES)

model: ## download the encoder into ./models (needs internet once)
	$(PY) -m retrieval.download_model

bench-encoder: ## encoder latency/throughput by batch size and threads
	$(PY) -m bench.bench_encoder --encoder $(ENCODER)

index: ## build index version V (WATERMARK=n optional)
	$(PY) -m retrieval.build_index --version $(V) --encoder $(ENCODER) --pad $(PAD) $(if $(WATERMARK),--watermark $(WATERMARK)) $(BUILD_ARGS)

validate: ## promotion gate for V (BASELINE=v1 optional)
	$(PY) -m retrieval.validate_index --version $(V) --encoder $(ENCODER) $(if $(BASELINE),--baseline $(BASELINE)) $(VALIDATE_ARGS)

bench-ann: ## HNSW recall vs latency/memory grid on the catalog embeddings
	$(PY) -m bench.bench_ann --vectors auto --encoder $(ENCODER) $(ANN_ARGS)

serve: ## run the service on :8000 against artifacts/indexes/V
	INDEX_DIR=artifacts/indexes/$(V) ENCODER_BACKEND=$(ENCODER) POD_NAME=laptop \
	  $(PY) -m uvicorn retrieval.server:app --host 0.0.0.0 --port 8000 --no-access-log --timeout-graceful-shutdown 20

load-local: ## open-loop load against localhost:8000 (RATE, DURATION, NAME)
	$(PY) -m loadgen.loadgen --target http://localhost:8000 --rate $(RATE) --duration $(DURATION) --out reports/$(NAME) $(LOAD_ARGS)
	$(PY) ops/plot_run.py reports/$(NAME)

sweep-local: ## capacity sweep against localhost:8000 (RATES)
	$(PY) -m loadgen.loadgen --target http://localhost:8000 --mode sweep --rates $(RATES) --step-duration 30 --out reports/$(NAME) $(LOAD_ARGS)
	$(PY) ops/plot_run.py reports/$(NAME)

# ---------------------------------------------------------------- images
image: ## build the service image (TAG=1.0)
	docker build -f deploy/Dockerfile --build-arg BUILD_ID=$(TAG) -t $(IMAGE) .

image-loadgen: ## build the load generator image
	docker build -f deploy/Dockerfile.loadgen -t $(LOADGEN_IMAGE) .

lock: ## freeze the image's resolved Python packages into requirements/serve.lock
	docker run --rm --entrypoint cat $(IMAGE) /opt/venv/requirements.freeze > requirements/serve.lock
	@echo "wrote requirements/serve.lock; rebuild with 'make image' to use it"

docker-run: ## run the image like a pod: 2 CPUs, memory cap, read-only rootfs, index mounted
	docker run --rm --name retrieval-local -p 8000:8000 --cpus 2 --memory $(or $(MEMORY),2g) \
	  --read-only --tmpfs /tmp:size=512m -v $(PWD)/artifacts/indexes:/index:ro \
	  -e INDEX_DIR=/index/$(V) -e ENCODER_BACKEND=$(ENCODER) -e POD_NAME=docker-local $(IMAGE)

# ---------------------------------------------------------------- cluster
cluster-up: ## create kind cluster + artifact server + Prometheus + metrics-server
	ops/kind_up.sh

cluster-down: ## delete the kind cluster
	kind delete cluster --name retrieval-lab

load-images: ## copy images into the kind nodes
	ops/load_images.sh $(IMAGE) $(LOADGEN_IMAGE)

deploy: ## deploy COLOR pinned to V (REPLICAS, IMAGE, DRAIN=on|off, FORCE=1)
	REPLICAS=$(REPLICAS) ops/deploy.sh $(COLOR) $(V)

promote: ## point Service retrieval at COLOR
	ops/promote.sh $(COLOR)

status: ## what is serving right now
	@ops/status.sh

loadtest: ## in-cluster constant load (NAME, RATE, DURATION, LOAD_ARGS)
	ops/loadtest.sh $(NAME) --rate $(RATE) --duration $(DURATION) $(LOAD_ARGS)

sweep: ## in-cluster capacity sweep (NAME, RATES, LOAD_ARGS)
	ops/loadtest.sh $(NAME) --mode sweep --rates $(RATES) --step-duration 30 $(LOAD_ARGS)

shadow: ## compare active color with the other color on the labeled queries
	ops/shadow.sh

hpa: ## autoscale COLOR on CPU (MIN, MAX, CPU_TARGET)
	python3 ops/render.py deploy/k8s/hpa.tmpl.yaml COLOR=$(COLOR) MIN_REPLICAS=$(or $(MIN),2) MAX_REPLICAS=$(or $(MAX),6) CPU_TARGET=$(or $(CPU_TARGET),60) | kubectl apply -f -

hpa-off: ## remove the autoscaler for COLOR
	kubectl -n search delete hpa retrieval-$(COLOR) --ignore-not-found

.PHONY: help venv test sizing data changes model bench-encoder index validate bench-ann serve load-local sweep-local \
	image image-loadgen lock docker-run cluster-up cluster-down load-images deploy promote status loadtest sweep shadow hpa hpa-off
