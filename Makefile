LATEXMK   := latexmk
SRCDIR    := paper/src
BUILDDIR  := build
TEX       := main.tex
JOBNAME   := main
PDF       := paper/$(BUILDDIR)/$(JOBNAME).pdf
VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
PYTEST    := $(VENV)/bin/pytest
NX        := npx nx

SWEEP_ENV_FILE ?= .env.sweep
TPU_CONF ?= tpu_orchestration/configs/v4_spot_us.conf

WANDB_ENTITY ?=
WANDB_PROJECT ?= capstone
SWEEP_ID ?=
LOCAL_TRAIN_ARGS ?= --algo ppo --total-timesteps 50000
LOCAL_BENCHMARK_ARGS ?= --tiers static,surge,linear,qtable,ppo --alpha-values 0.0,0.3 --episodes 3 --total-timesteps 3000 --max-steps 40 --device cpu
SIMPLE_BENCHMARK_ARGS ?= --tiers qtable,ppo,dqn,a2c --alpha-values 0.0,0.15,0.3,0.45,0.6 --episodes 8 --total-timesteps 8000 --max-steps 40 --device cpu
BENCHMARK_AGENT_ARGS ?=
AGENT_COUNT ?= 0

WHOCLICKED_REPO ?= velocitatem/whoclickedit
WHOCLICKED_CSV ?= experiments/exports/whoclicked.csv
WHOCLICKED_CARD ?= experiments/exports/whoclicked_dataset_card.md
WHOCLICKED_CSV_PATH_IN_REPO ?= whoclicked.csv
WHOCLICKED_CARD_PATH_IN_REPO ?= README.md
WHOCLICKED_DATASET_MESSAGE ?= Update flattened whoclickedit dataset
WHOCLICKED_CARD_MESSAGE ?= Update dataset card for whoclickedit

REPO_URL ?=
BRANCH ?= main
WORKDIR ?= $(HOME)/PHANTOM-agent
AGENT_LOOP ?= 1
RETRY_SECONDS ?= 20

TRAIN_IMAGE_REF := us-central1-docker.pkg.dev/phantom-trc/phantom/phantom-trainer

SWEEP_ENV_LOAD = set -a; [ -f "$(SWEEP_ENV_FILE)" ] && . "$(SWEEP_ENV_FILE)" || true; set +a

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "pdf.build pdf.watch pdf.clean pdf.genpop pdf.genpop.watch pdf.arxiv | test.backend test.e2e test.all | web.dev | install | train | benchmark | benchmark.simple | benchmark.agent | train.agent | train.bootstrap | stats.lines | docs.platform | manim.defense manim.defense.hq manim.render manim.render.full manim.render.poster manim.render.appendix manim.render.all"
	@echo "backend.server backend.provider backend.worker | platform.up platform.down platform.logs | docker.train.publish"
	@echo "data.pull data.push data.whoclicked.publish | study.margin-erosion study.margin-erosion.quick study.margin-erosion.plot"
	@echo "tpu.ray.bootstrap tpu.ray.deps tpu.ray.verify tpu.ray.teardown"
	@echo ""
	@echo "Build general public version:"
	@echo "  make pdf.genpop"
	@echo ""
	@echo "Local wandb run:"
	@echo "  make train LOCAL_TRAIN_ARGS='--algo ppo --total-timesteps 50000'"
	@echo ""
	@echo "Local benchmark run:"
	@echo "  make benchmark LOCAL_BENCHMARK_ARGS='--tiers static,surge,linear --alpha-values 0.0,0.3 --episodes 3 --no-wandb'"
	@echo ""
	@echo "Simple benchmark run (.env.sweep defaults, robust+no_robust compare by default):"
	@echo "  make benchmark.simple"
	@echo ""
	@echo "Local sweep agent from this repo:"
	@echo "  make train.agent SWEEP_ID=entity/project/id AGENT_COUNT=5"
	@echo ""
	@echo "Bootstrap private repo worker from anywhere:"
	@echo "  make train.bootstrap REPO_URL=https://github.com/org/repo.git BRANCH=main SWEEP_ID=entity/project/id"
	@echo ""
	@echo "Bootstrap Ray on TPU slice from config:"
	@echo "  make tpu.ray.bootstrap TPU_CONF=tpu_orchestration/configs/v4_spot_us.conf"
	@echo ""
	@echo "Publish whoclickedit dataset + card:"
	@echo "  make data.whoclicked.publish HF_TOKEN=... WHOCLICKED_REPO=velocitatem/whoclickedit"
	@echo ""
	@echo "Config source: $(SWEEP_ENV_FILE) (auto-loaded)"

$(BUILDDIR):
	mkdir -p paper/$(BUILDDIR)

.PHONY: pdf.build
pdf.build:
	@$(NX) run paper:build

.PHONY: pdf.watch
pdf.watch:
	@$(NX) run paper:watch

.PHONY: pdf.clean
pdf.clean:
	@$(NX) run paper:clean

.PHONY: pdf.genpop
pdf.genpop:
	@bash scripts/nx_paper.sh build-genpop

.PHONY: pdf.genpop.watch
pdf.genpop.watch:
	@bash scripts/nx_paper.sh watch-genpop

.PHONY: pdf.arxiv
pdf.arxiv:
	@bash scripts/nx_paper.sh build-arxiv

.PHONY: test.backend
test.backend:
	@$(NX) run research:test

.PHONY: test.e2e
test.e2e:
	@$(NX) run e2e:test

.PHONY: test.all
test.all:
	@$(NX) run-many -t test --projects=research,e2e --parallel=1

.PHONY: web.dev
web.dev:
	@$(NX) run web:dev

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install
install:
	@$(NX) run research:install

.PHONY: train
train:
	@WANDB_ENTITY="$(WANDB_ENTITY)" WANDB_PROJECT="$(WANDB_PROJECT)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" LOCAL_TRAIN_ARGS="$(LOCAL_TRAIN_ARGS)" $(NX) run research:train

.PHONY: benchmark
benchmark:
	@WANDB_ENTITY="$(WANDB_ENTITY)" WANDB_PROJECT="$(WANDB_PROJECT)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" LOCAL_BENCHMARK_ARGS="$(LOCAL_BENCHMARK_ARGS)" $(NX) run research:benchmark

.PHONY: benchmark.simple
benchmark.simple:
	@WANDB_ENTITY="$(WANDB_ENTITY)" WANDB_PROJECT="$(WANDB_PROJECT)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" SIMPLE_BENCHMARK_ARGS="$(SIMPLE_BENCHMARK_ARGS)" PHANTOM_BENCHMARK_COMPARE_ROBUST="$(PHANTOM_BENCHMARK_COMPARE_ROBUST)" $(NX) run research:benchmark-simple

.PHONY: benchmark.agent
benchmark.agent:
	@WANDB_ENTITY="$(WANDB_ENTITY)" WANDB_PROJECT="$(WANDB_PROJECT)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" SWEEP_ID="$(SWEEP_ID)" AGENT_COUNT="$(AGENT_COUNT)" BENCHMARK_AGENT_ARGS="$(BENCHMARK_AGENT_ARGS)" $(NX) run research:benchmark-agent

.PHONY: train.agent
train.agent:
	@WANDB_ENTITY="$(WANDB_ENTITY)" WANDB_PROJECT="$(WANDB_PROJECT)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" SWEEP_ID="$(SWEEP_ID)" AGENT_COUNT="$(AGENT_COUNT)" $(NX) run research:train-agent

.PHONY: train.bootstrap
train.bootstrap:
	@WANDB_ENTITY="$(WANDB_ENTITY)" WANDB_PROJECT="$(WANDB_PROJECT)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" REPO_URL="$(REPO_URL)" BRANCH="$(BRANCH)" WORKDIR="$(WORKDIR)" SWEEP_ID="$(SWEEP_ID)" AGENT_COUNT="$(AGENT_COUNT)" AGENT_LOOP="$(AGENT_LOOP)" RETRY_SECONDS="$(RETRY_SECONDS)" $(NX) run research:train-bootstrap

.PHONY: tpu.ray.bootstrap tpu.ray.deps tpu.ray.verify tpu.ray.teardown
tpu.ray.bootstrap:
	@TPU_CONF="$(TPU_CONF)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" $(NX) run research:tpu-ray-bootstrap

tpu.ray.deps:
	@TPU_CONF="$(TPU_CONF)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" $(NX) run research:tpu-ray-deps

tpu.ray.verify:
	@TPU_CONF="$(TPU_CONF)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" $(NX) run research:tpu-ray-verify

tpu.ray.teardown:
	@TPU_CONF="$(TPU_CONF)" SWEEP_ENV_FILE="$(SWEEP_ENV_FILE)" $(NX) run research:tpu-ray-teardown

.PHONY: data.pull data.push
data.pull:
	python scripts/hf_data.py pull

data.push:
	python scripts/hf_data.py push

.PHONY: data.whoclicked.publish
data.whoclicked.publish:
	@HF_TOKEN="$(HF_TOKEN)" WHOCLICKED_REPO="$(WHOCLICKED_REPO)" WHOCLICKED_CSV="$(WHOCLICKED_CSV)" WHOCLICKED_CARD="$(WHOCLICKED_CARD)" WHOCLICKED_CSV_PATH_IN_REPO="$(WHOCLICKED_CSV_PATH_IN_REPO)" WHOCLICKED_CARD_PATH_IN_REPO="$(WHOCLICKED_CARD_PATH_IN_REPO)" WHOCLICKED_DATASET_MESSAGE="$(WHOCLICKED_DATASET_MESSAGE)" WHOCLICKED_CARD_MESSAGE="$(WHOCLICKED_CARD_MESSAGE)" $(NX) run research:whoclicked-publish

.PHONY: stats.lines
stats.lines:
	@$(NX) run research:stats

.PHONY: study.margin-erosion
study.margin-erosion:
	python -m engine.studies.margin_erosion_alpha

.PHONY: study.margin-erosion.quick
study.margin-erosion.quick:
	python -m engine.studies.margin_erosion_alpha --quick

DOCS_VENV ?= docs/.venv
DOCS_MKDOCS := $(DOCS_VENV)/bin/mkdocs
DOCS_PIP := $(DOCS_VENV)/bin/pip

.PHONY: docs.platform
docs.platform: $(DOCS_VENV)
	$(DOCS_MKDOCS) build -f docs/mkdocs.yml

$(DOCS_VENV):
	python3 -m venv $(DOCS_VENV)
	$(DOCS_PIP) install --upgrade pip
	$(DOCS_PIP) install -r docs/requirements.txt

.PHONY: wordcount
wordcount:
	@$(NX) run paper:wordcount

.PHONY: docker.train.publish
docker.train.publish:
	@TRAIN_IMAGE_REF="$(TRAIN_IMAGE_REF)" $(NX) run research:docker-train-publish

.PHONY: backend.server backend.provider backend.worker platform.up platform.down platform.logs
backend.server:
	@$(NX) run backend-server:dev

backend.provider:
	@$(NX) run pricing-provider:dev

backend.worker:
	@$(NX) run backend-worker:dev

platform.up:
	@$(NX) run platform:up

platform.down:
	@$(NX) run platform:down

platform.logs:
	@$(NX) run platform:logs

.PHONY: pdf clean watch run.webapp test count-lines all
pdf:
	@$(NX) run paper:build

clean:
	@$(NX) run paper:clean

watch:
	@$(NX) run paper:watch

run.webapp:
	@$(NX) run web:dev

test:
	@$(NX) run research:test

count-lines:
	@$(NX) run research:stats

all:
	@$(NX) run paper:build

.PHONY: manim.defense manim.defense.hq manim.render manim.render.full manim.render.poster manim.render.appendix manim.render.all
# Main defense reel (paper/defense/manim/render_defense); uses paper/defense/.venv when present
manim.defense:
	@cd paper/defense/manim && ./render_defense full

manim.defense.hq:
	@cd paper/defense/manim && ./render_defense full --quality qh

manim.render:
	@$(NX) run manim:render

manim.render.full:
	@$(NX) run manim:render-full

manim.render.poster:
	@$(NX) run manim:render-poster

manim.render.appendix:
	@$(NX) run manim:render-appendix

manim.render.all:
	@$(NX) run manim:render-all
