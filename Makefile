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

SWEEP_ENV_FILE ?= .env.sweep

WANDB_ENTITY ?=
WANDB_PROJECT ?= phantom-pricing
SWEEP_ID ?=
LOCAL_TRAIN_ARGS ?= --algo ppo --total-timesteps 50000
AGENT_COUNT ?= 0

REPO_URL ?=
BRANCH ?= main
WORKDIR ?= $(HOME)/PHANTOM-agent
AGENT_LOOP ?= 1
RETRY_SECONDS ?= 20

TRAIN_IMAGE_REF := us-central1-docker.pkg.dev/phantom-trc/phantom/phantom-trainer
TPU_NAME ?=
TPU_ZONE ?= us-central2-b

SWEEP_ENV_LOAD = set -a; [ -f "$(SWEEP_ENV_FILE)" ] && . "$(SWEEP_ENV_FILE)" || true; set +a

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "pdf.build pdf.watch pdf.clean | test.backend test.e2e test.all | web.dev | install | train | train.agent | train.bootstrap | train.tpu.pod | stats.lines"
	@echo "docker.train.publish"
	@echo ""
	@echo "Local wandb run:"
	@echo "  make train LOCAL_TRAIN_ARGS='--algo ppo --total-timesteps 50000'"
	@echo ""
	@echo "Local sweep agent from this repo:"
	@echo "  make train.agent SWEEP_ID=entity/project/id AGENT_COUNT=5"
	@echo ""
	@echo "Bootstrap private repo worker from anywhere:"
	@echo "  make train.bootstrap REPO_URL=https://github.com/org/repo.git BRANCH=main SWEEP_ID=entity/project/id"
	@echo ""
	@echo "Config source: $(SWEEP_ENV_FILE) (auto-loaded)"

$(BUILDDIR):
	mkdir -p paper/$(BUILDDIR)

.PHONY: pdf.build
pdf.build: $(BUILDDIR)
	@bash paper/concat_code.sh
	@cd $(SRCDIR) && \
	$(LATEXMK) -pdf -jobname=$(JOBNAME) -f \
		-interaction=nonstopmode -file-line-error \
		-r ../.latexmkrc \
		-outdir=../$(BUILDDIR) $(TEX)

.PHONY: pdf.watch
pdf.watch: $(BUILDDIR)
	@cd $(SRCDIR) && \
	$(LATEXMK) -pvc -pdf -jobname=$(JOBNAME) -f \
		-interaction=nonstopmode -file-line-error \
		-r ../.latexmkrc \
		-outdir=../$(BUILDDIR) $(TEX)

.PHONY: pdf.clean
pdf.clean:
	@cd $(SRCDIR) && \
	$(LATEXMK) -C -jobname=$(JOBNAME) -outdir=../$(BUILDDIR) || true
	rm -rf paper/$(BUILDDIR)/*

.PHONY: test.backend
test.backend: $(VENV)
	$(PYTEST) -v

.PHONY: test.e2e
test.e2e:
	@cd tests/e2e && npm install
	@cd tests/e2e && npx playwright install chromium
	@test -f tests/e2e/.env || cp tests/e2e/.env.example tests/e2e/.env
	@timeout 30 bash -c 'until curl -sf http://localhost:5000/health > /dev/null 2>&1; do sleep 1; done' || (echo "Backend not ready" && exit 1)
	@timeout 30 bash -c 'until curl -sf http://localhost:3000 > /dev/null 2>&1; do sleep 1; done' || (echo "Web app not ready" && exit 1)
	@timeout 30 bash -c 'until curl -sf http://localhost:8085/health > /dev/null 2>&1; do sleep 1; done' || (echo "Airflow not ready" && exit 1)
	@cd tests/e2e && npm test

.PHONY: test.all
test.all: test.backend test.e2e

.PHONY: web.dev
web.dev:
	@cd web && npm install && npm run dev

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install
install: $(VENV)
	$(PIP) install -r requirements.txt

.PHONY: train
train: install
	@$(SWEEP_ENV_LOAD); test -n "$$WANDB_API_KEY" || (echo "WANDB_API_KEY required — set it in $(SWEEP_ENV_FILE)" && exit 1)
	@$(SWEEP_ENV_LOAD); WANDB_API_KEY="$$WANDB_API_KEY" WANDB_ENTITY="$(WANDB_ENTITY)" WANDB_PROJECT="$(WANDB_PROJECT)" \
		$(PYTHON) -m engine.train $(LOCAL_TRAIN_ARGS)

.PHONY: train.agent
train.agent: install
	@$(SWEEP_ENV_LOAD); test -n "$$WANDB_API_KEY" || (echo "WANDB_API_KEY required — set it in $(SWEEP_ENV_FILE)" && exit 1)
	@test -n "$(SWEEP_ID)" || (echo "SWEEP_ID required, e.g. SWEEP_ID=entity/project/id" && exit 1)
	@$(SWEEP_ENV_LOAD); WANDB_API_KEY="$$WANDB_API_KEY" WANDB_ENTITY="$(WANDB_ENTITY)" WANDB_PROJECT="$(WANDB_PROJECT)" \
		$(PYTHON) -m engine.train --sweep-agent --sweep-id "$(SWEEP_ID)" \
		$(if $(filter-out 0,$(AGENT_COUNT)),--count $(AGENT_COUNT),)

.PHONY: train.bootstrap
train.bootstrap:
	@$(SWEEP_ENV_LOAD); test -n "$$WANDB_API_KEY" || (echo "WANDB_API_KEY required — set it in $(SWEEP_ENV_FILE)" && exit 1)
	@$(SWEEP_ENV_LOAD); test -n "$$GITHUB_TOKEN" || (echo "GITHUB_TOKEN required — set it in $(SWEEP_ENV_FILE)" && exit 1)
	@test -n "$(REPO_URL)" || (echo "REPO_URL required, e.g. REPO_URL=https://github.com/org/repo.git" && exit 1)
	@test -n "$(SWEEP_ID)" || (echo "SWEEP_ID required, e.g. SWEEP_ID=entity/project/id" && exit 1)
	@$(SWEEP_ENV_LOAD); \
		WANDB_API_KEY="$$WANDB_API_KEY" \
		WANDB_ENTITY="$(WANDB_ENTITY)" \
		WANDB_PROJECT="$(WANDB_PROJECT)" \
		GITHUB_TOKEN="$$GITHUB_TOKEN" \
		REPO_URL="$(REPO_URL)" \
		BRANCH="$(BRANCH)" \
		WORKDIR="$(WORKDIR)" \
		SWEEP_ID="$(SWEEP_ID)" \
		AGENT_COUNT="$(AGENT_COUNT)" \
		AGENT_LOOP="$(AGENT_LOOP)" \
		RETRY_SECONDS="$(RETRY_SECONDS)" \
		bash scripts/wandb_agent_bootstrap.sh

.PHONY: stats.lines
stats.lines:
	@find . \( -path '*/node_modules' -o -path '*/.venv' -o -path '*/venv' \) -prune -o \
	\( -name "*.ts" -o -name "*.py" \) -type f -print0 | xargs -0 cat | wc -l

.PHONY: wordcount
wordcount:
	@echo "Counting words in main text (excluding appendix)..."
	@texcount -nosub -total -sum -1 \
		$(SRCDIR)/chapters/01-intro.tex \
		$(SRCDIR)/chapters/02-literature-review.tex \
		$(SRCDIR)/chapters/03-methodology.tex \
		$(SRCDIR)/chapters/04-results.tex \
		$(SRCDIR)/chapters/05-discussion.tex \
		$(SRCDIR)/chapters/06-conclusion.tex

.PHONY: docker.train.publish
docker.train.publish:
	docker build -f docker/Trainer.dockerfile --target gpu -t $(TRAIN_IMAGE_REF):gpu-latest .
	docker push $(TRAIN_IMAGE_REF):gpu-latest
	docker build -f docker/Trainer.dockerfile --target tpu -t $(TRAIN_IMAGE_REF):tpu-latest .
	docker push $(TRAIN_IMAGE_REF):tpu-latest

.PHONY: train.tpu.pod
train.tpu.pod:
	@test -n "$(TPU_NAME)"  || (echo "TPU_NAME required, e.g. TPU_NAME=TPUlong" && exit 1)
	@test -n "$(SWEEP_ID)"  || (echo "SWEEP_ID required, e.g. SWEEP_ID=entity/project/id" && exit 1)
	@$(SWEEP_ENV_LOAD); test -n "$$WANDB_API_KEY" || (echo "WANDB_API_KEY required — set it in $(SWEEP_ENV_FILE)" && exit 1)
	gcloud compute tpus tpu-vm scp scripts/tpu_pod_run.sh $(TPU_NAME):/tmp/tpu_pod_run.sh \
		--zone=$(TPU_ZONE) --project=phantom-trc --worker=all
	@$(SWEEP_ENV_LOAD); \
		gcloud compute tpus tpu-vm ssh $(TPU_NAME) \
		--zone=$(TPU_ZONE) --project=phantom-trc --worker=all \
		--command="WANDB_API_KEY='$$WANDB_API_KEY' SWEEP_ID='$(SWEEP_ID)' AGENT_COUNT='$(AGENT_COUNT)' sh /tmp/tpu_pod_run.sh"

.PHONY: pdf clean watch run.webapp test count-lines all
pdf: pdf.build
clean: pdf.clean
watch: pdf.watch
run.webapp: web.dev
test: test.backend
count-lines: stats.lines
all: pdf.build
