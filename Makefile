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
TPU_NAME  ?= phantom-tpu
TPU_ZONE  ?= us-central2-b
TPU_TYPE  ?= v4-32
TPU_RUNTIME ?= tpu-vm-v4-base
TPU_PROJECT ?= phantom-trc
TPU_NETWORK ?= tpu-network
TPU_SUBNETWORK ?= tpu-network
TPU_USE_SPOT ?= 0
TPU_EXTRA_CREATE_FLAGS ?=
TPU_WORKDIR ?= ~/PHANTOM
TPU_SYNC_PATHS ?= engine lib requirements.txt Makefile .env
TPU_TRAIN_ARGS ?= --algo ppo --jax --total-timesteps 20000
TPU_JAX_WHEEL_URL ?= https://storage.googleapis.com/jax-releases/libtpu_releases.html
TPU_VENV ?= .venv-tpu
TPU_TRAIN_ENV ?= PHANTOM_USE_JAX=1 WANDB_MODE=online
SWEEP_ID ?=
SWEEP_COUNT ?= 5
QUEUE_SCRIPT ?= scripts/queue_sweep.sh
TPU_QUEUE_TYPE ?=
TPU_QUEUE_ZONES ?= europe-west4-a us-central2-b us-central1-a us-east1-d europe-west4-b
TPU_QUEUE_REUSE_EXISTING ?= 1
TPU_QUEUE_KEEP_ALIVE ?= 1
TPU_QUEUE_STRICT_QUOTA ?= 0
TPU_QUEUE_DOWNSHIFT_ON_QUOTA ?= 1
TPU_QUEUE_FILTER_ZONE ?=
TPU_QUEUE_FILTER_TYPE ?=
TPU_QUEUE_EXECUTION_MODE ?= venv
TPU_QUEUE_SYNC_METHOD ?= tar
TPU_QUEUE_SKIP_SYNC ?= 0
TPU_QUEUE_DOCKER_IMAGE ?=
TPU_QUEUE_DOCKER_PULL ?= 1
TPU_QUEUE_DOCKER_AUTO_INSTALL ?= 1
TPU_QUEUE_SSH_BATCH_MODE ?= 1
TPU_QUEUE_SSH_CONNECT_TIMEOUT ?= 12
TPU_QUEUE_SSH_KEY_FILE ?= $(HOME)/.ssh/google_compute_engine
TPU_QUEUE_REQUIRE_SSH_AGENT ?= 1
TPU_QUEUE_AUTO_SSH_ADD ?= 1
TPU_SPOT_FLAG := $(if $(filter 1 true TRUE yes YES,$(TPU_USE_SPOT)),--spot,)
TPU_CREATE_CMD = gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm create "$(TPU_NAME)" --zone="$(TPU_ZONE)" --accelerator-type="$(TPU_TYPE)" --version="$(TPU_RUNTIME)" --network="$(TPU_NETWORK)" --subnetwork="$(TPU_SUBNETWORK)" $(TPU_SPOT_FLAG) $(TPU_EXTRA_CREATE_FLAGS)

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "pdf.build pdf.watch pdf.clean | test.backend test.e2e test.all | web.dev | install | stats.lines | tpu.* | tpu.queue.*"
	@echo "TPU presets: tpu.create.v4.ondemand | tpu.create.v4.spot"
	@echo "Queued sweep: SWEEP_ID=entity/project/id make tpu.queue.sweep"
	@echo "Queued sweep filters: TPU_QUEUE_FILTER_TYPE=v6e TPU_QUEUE_FILTER_ZONE=europe-west4-a"
	@echo "Docker queue: make tpu.queue.sweep.docker TPU_QUEUE_DOCKER_IMAGE=gcr.io/<project>/<image>:tag"
	@echo "Docker queue without sync: add TPU_QUEUE_SKIP_SYNC=1"
	@echo "If SSH key is encrypted: run ssh-add ~/.ssh/google_compute_engine first"

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

.PHONY: tpu.setup
tpu.setup:
	@command -v gcloud >/dev/null 2>&1 || (echo "gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install" && exit 1)
	@gcloud auth login --update-adc
	@gcloud auth application-default login
	@gcloud config set project "$(TPU_PROJECT)"

.PHONY: tpu.check.zone
tpu.check.zone:
	@case "$(TPU_ZONE)" in \
		europe-west4-a|us-central2-b|us-central1-a|us-east1-d|europe-west4-b) ;; \
		*) echo "Unsupported TPU_ZONE='$(TPU_ZONE)'. Allowed zones: europe-west4-a us-central2-b us-central1-a us-east1-d europe-west4-b"; exit 1 ;; \
	 esac

.PHONY: tpu.create.v4.ondemand
tpu.create.v4.ondemand:
	$(MAKE) tpu.create TPU_ZONE=us-central2-b TPU_TYPE=v4-32 TPU_USE_SPOT=0 TPU_SUBNETWORK=tpu-network

.PHONY: tpu.create.v4.spot
tpu.create.v4.spot:
	$(MAKE) tpu.create TPU_ZONE=us-central2-b TPU_TYPE=v4-32 TPU_USE_SPOT=1 TPU_SUBNETWORK=tpu-network

.PHONY: tpu.create
tpu.create: tpu.check.zone
	@if gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm describe "$(TPU_NAME)" --zone="$(TPU_ZONE)" >/dev/null 2>&1; then \
		STATE=$$(gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm describe "$(TPU_NAME)" --zone="$(TPU_ZONE)" --format='value(state)'); \
		echo "TPU VM $(TPU_NAME) already exists in $(TPU_ZONE) with state=$$STATE, skipping create"; \
	else \
		$(TPU_CREATE_CMD); \
	fi

.PHONY: tpu.ensure
tpu.ensure: tpu.check.zone
	@set -e; \
	STATE=$$(gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm describe "$(TPU_NAME)" --zone="$(TPU_ZONE)" --format='value(state)' 2>/dev/null || true); \
	if [ -z "$$STATE" ]; then \
		echo "TPU VM $(TPU_NAME) not found in $(TPU_ZONE), creating"; \
		$(TPU_CREATE_CMD); \
	elif [ "$$STATE" = "READY" ]; then \
		echo "TPU VM $(TPU_NAME) is READY"; \
	elif [ "$$STATE" = "PREEMPTED" ] || [ "$$STATE" = "TERMINATED" ] || [ "$$STATE" = "FAILED" ]; then \
		echo "TPU VM $(TPU_NAME) is in terminal state $$STATE, recreating"; \
		gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm delete "$(TPU_NAME)" --zone="$(TPU_ZONE)" --quiet || true; \
		$(TPU_CREATE_CMD); \
	else \
		echo "TPU VM $(TPU_NAME) is in state $$STATE; wait or recreate manually"; \
		exit 1; \
	fi

.PHONY: tpu.status
tpu.status:
	gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm describe "$(TPU_NAME)" --zone="$(TPU_ZONE)"

.PHONY: tpu.ssh
tpu.ssh:
	gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm ssh "$(TPU_NAME)" --zone="$(TPU_ZONE)"

.PHONY: tpu.prepare
tpu.prepare: tpu.ensure
	gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm ssh "$(TPU_NAME)" --zone="$(TPU_ZONE)" --command "mkdir -p $(TPU_WORKDIR)"

.PHONY: tpu.deploy
tpu.deploy: tpu.prepare
	@for p in $(TPU_SYNC_PATHS); do \
		if [ ! -e "$$p" ]; then continue; fi; \
		if [ -d "$$p" ]; then \
			gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm scp --recurse "$$p" "$(TPU_NAME):$(TPU_WORKDIR)/$$p" --zone="$(TPU_ZONE)"; \
		else \
			gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm scp "$$p" "$(TPU_NAME):$(TPU_WORKDIR)/$$p" --zone="$(TPU_ZONE)"; \
		fi; \
	done

.PHONY: tpu.install
tpu.install: tpu.ensure
	gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm ssh "$(TPU_NAME)" --zone="$(TPU_ZONE)" --command 'cd $(TPU_WORKDIR) && PYBIN=$$(command -v python3.11 || command -v python3.10 || command -v python3) && $$PYBIN -m venv $(TPU_VENV) && $(TPU_VENV)/bin/pip install --upgrade pip setuptools wheel && $(TPU_VENV)/bin/pip install -r requirements.txt && $(TPU_VENV)/bin/pip install -r engine/jax/requirements.txt && $(TPU_VENV)/bin/pip install "jax[tpu]" -f $(TPU_JAX_WHEEL_URL)'

.PHONY: tpu.check.remote
tpu.check.remote: tpu.ensure
	gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm ssh "$(TPU_NAME)" --zone="$(TPU_ZONE)" --command 'set -e; mkdir -p $(TPU_WORKDIR); cd $(TPU_WORKDIR); test -f engine/train.py || (echo "Missing code on TPU VM. Run: make tpu.deploy" && exit 2); test -x $(TPU_VENV)/bin/python || (echo "Missing TPU venv. Run: make tpu.install" && exit 3)'

.PHONY: tpu.train
tpu.train: tpu.check.remote
	gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm ssh "$(TPU_NAME)" --zone="$(TPU_ZONE)" --command 'cd $(TPU_WORKDIR) && if [ -f .env ]; then set -a && . ./.env && set +a; fi && $(TPU_TRAIN_ENV) $(TPU_VENV)/bin/python -m engine.train $(TPU_TRAIN_ARGS)'

.PHONY: tpu.bootstrap
tpu.bootstrap: tpu.ensure tpu.deploy tpu.install

.PHONY: tpu.delete
tpu.delete:
	gcloud --project="$(TPU_PROJECT)" compute tpus tpu-vm delete "$(TPU_NAME)" --zone="$(TPU_ZONE)" --quiet

.PHONY: tpu.queue.sweep
tpu.queue.sweep:
	@set -e; \
	test -n "$(SWEEP_ID)" || (echo "SWEEP_ID is required, e.g. SWEEP_ID=entity/project/id" && exit 1); \
	test -n "$$WANDB_API_KEY" || (echo "WANDB_API_KEY is required in your shell" && exit 1); \
	if [ "$(TPU_QUEUE_AUTO_SSH_ADD)" = "1" ] && [ "$(TPU_QUEUE_SSH_BATCH_MODE)" != "0" ] && command -v ssh-add >/dev/null 2>&1 && [ -f "$(TPU_QUEUE_SSH_KEY_FILE)" ]; then \
		if ! ssh-add -l >/dev/null 2>&1; then \
			if [ -z "$$SSH_AUTH_SOCK" ] && command -v ssh-agent >/dev/null 2>&1; then eval "$$(ssh-agent -s)" >/dev/null; fi; \
			ssh-add "$(TPU_QUEUE_SSH_KEY_FILE)"; \
		fi; \
	fi; \
	AGENT_COUNT="$(SWEEP_COUNT)" PROJECT_ID="$(TPU_PROJECT)" TPU_NETWORK="$(TPU_NETWORK)" TPU_SUBNETWORK="$(TPU_SUBNETWORK)" TPU_REUSE_EXISTING="$(TPU_QUEUE_REUSE_EXISTING)" TPU_KEEP_ALIVE="$(TPU_QUEUE_KEEP_ALIVE)" TPU_STRICT_QUOTA="$(TPU_QUEUE_STRICT_QUOTA)" TPU_DOWNSHIFT_ON_QUOTA="$(TPU_QUEUE_DOWNSHIFT_ON_QUOTA)" TPU_EXECUTION_MODE="$(TPU_QUEUE_EXECUTION_MODE)" TPU_SYNC_METHOD="$(TPU_QUEUE_SYNC_METHOD)" TPU_SKIP_SYNC="$(TPU_QUEUE_SKIP_SYNC)" TPU_DOCKER_IMAGE="$(TPU_QUEUE_DOCKER_IMAGE)" TPU_DOCKER_PULL="$(TPU_QUEUE_DOCKER_PULL)" TPU_DOCKER_AUTO_INSTALL="$(TPU_QUEUE_DOCKER_AUTO_INSTALL)" TPU_SSH_BATCH_MODE="$(TPU_QUEUE_SSH_BATCH_MODE)" TPU_SSH_CONNECT_TIMEOUT="$(TPU_QUEUE_SSH_CONNECT_TIMEOUT)" TPU_SSH_KEY_FILE="$(TPU_QUEUE_SSH_KEY_FILE)" TPU_REQUIRE_SSH_AGENT="$(TPU_QUEUE_REQUIRE_SSH_AGENT)" TPU_QUEUE_FILTER_ZONE="$(TPU_QUEUE_FILTER_ZONE)" TPU_QUEUE_FILTER_TYPE="$(TPU_QUEUE_FILTER_TYPE)" WANDB_API_KEY="$$WANDB_API_KEY" "$(QUEUE_SCRIPT)" "$(SWEEP_ID)"

.PHONY: tpu.queue.worker
tpu.queue.worker:
	@set -e; \
	test -n "$(SWEEP_ID)" || (echo "SWEEP_ID is required, e.g. SWEEP_ID=entity/project/id" && exit 1); \
	test -n "$$WANDB_API_KEY" || (echo "WANDB_API_KEY is required in your shell" && exit 1); \
	if [ "$(TPU_QUEUE_AUTO_SSH_ADD)" = "1" ] && [ "$(TPU_QUEUE_SSH_BATCH_MODE)" != "0" ] && command -v ssh-add >/dev/null 2>&1 && [ -f "$(TPU_QUEUE_SSH_KEY_FILE)" ]; then \
		if ! ssh-add -l >/dev/null 2>&1; then \
			if [ -z "$$SSH_AUTH_SOCK" ] && command -v ssh-agent >/dev/null 2>&1; then eval "$$(ssh-agent -s)" >/dev/null; fi; \
			ssh-add "$(TPU_QUEUE_SSH_KEY_FILE)"; \
		fi; \
	fi; \
	AGENT_COUNT="$(SWEEP_COUNT)" PROJECT_ID="$(TPU_PROJECT)" TPU_NETWORK="$(TPU_NETWORK)" TPU_SUBNETWORK="$(TPU_SUBNETWORK)" TPU_REUSE_EXISTING="$(TPU_QUEUE_REUSE_EXISTING)" TPU_KEEP_ALIVE="$(TPU_QUEUE_KEEP_ALIVE)" TPU_STRICT_QUOTA="$(TPU_QUEUE_STRICT_QUOTA)" TPU_DOWNSHIFT_ON_QUOTA="$(TPU_QUEUE_DOWNSHIFT_ON_QUOTA)" TPU_EXECUTION_MODE="$(TPU_QUEUE_EXECUTION_MODE)" TPU_SYNC_METHOD="$(TPU_QUEUE_SYNC_METHOD)" TPU_SKIP_SYNC="$(TPU_QUEUE_SKIP_SYNC)" TPU_DOCKER_IMAGE="$(TPU_QUEUE_DOCKER_IMAGE)" TPU_DOCKER_PULL="$(TPU_QUEUE_DOCKER_PULL)" TPU_DOCKER_AUTO_INSTALL="$(TPU_QUEUE_DOCKER_AUTO_INSTALL)" TPU_SSH_BATCH_MODE="$(TPU_QUEUE_SSH_BATCH_MODE)" TPU_SSH_CONNECT_TIMEOUT="$(TPU_QUEUE_SSH_CONNECT_TIMEOUT)" TPU_SSH_KEY_FILE="$(TPU_QUEUE_SSH_KEY_FILE)" TPU_REQUIRE_SSH_AGENT="$(TPU_QUEUE_REQUIRE_SSH_AGENT)" TPU_QUEUE_FILTER_ZONE="$(TPU_ZONE)" TPU_QUEUE_FILTER_TYPE="$(TPU_QUEUE_TYPE)" WANDB_API_KEY="$$WANDB_API_KEY" "$(QUEUE_SCRIPT)" "$(SWEEP_ID)"

.PHONY: tpu.queue.sweep.docker
tpu.queue.sweep.docker:
	@test -n "$(TPU_QUEUE_DOCKER_IMAGE)" || (echo "TPU_QUEUE_DOCKER_IMAGE is required" && exit 1)
	@$(MAKE) tpu.queue.sweep TPU_QUEUE_EXECUTION_MODE=docker

.PHONY: tpu.queue.worker.docker
tpu.queue.worker.docker:
	@test -n "$(TPU_QUEUE_DOCKER_IMAGE)" || (echo "TPU_QUEUE_DOCKER_IMAGE is required" && exit 1)
	@$(MAKE) tpu.queue.worker TPU_QUEUE_EXECUTION_MODE=docker

.PHONY: tpu.queue.docker.build
tpu.queue.docker.build:
	@test -n "$(TPU_QUEUE_DOCKER_IMAGE)" || (echo "TPU_QUEUE_DOCKER_IMAGE is required" && exit 1)
	docker build -f docker/TPUSweep.Dockerfile -t "$(TPU_QUEUE_DOCKER_IMAGE)" .

.PHONY: tpu.queue.docker.push
tpu.queue.docker.push:
	@test -n "$(TPU_QUEUE_DOCKER_IMAGE)" || (echo "TPU_QUEUE_DOCKER_IMAGE is required" && exit 1)
	docker push "$(TPU_QUEUE_DOCKER_IMAGE)"

.PHONY: tpu.queue.status
tpu.queue.status:
	@set -e; \
	if gcloud compute tpus queued-resources list --help >/dev/null 2>&1; then \
		QCMD='gcloud --project=$(TPU_PROJECT) compute tpus queued-resources'; \
	else \
		QCMD='gcloud --project=$(TPU_PROJECT) alpha compute tpus queued-resources'; \
	fi; \
	for ZONE in $(TPU_QUEUE_ZONES); do \
		echo "--- $$ZONE ---"; \
		if ! $$QCMD list --zone="$$ZONE"; then \
			echo "Skipping $$ZONE (unavailable or no permission)"; \
		fi; \
	done

.PHONY: tpu.queue.clean
tpu.queue.clean:
	@set -e; \
	if gcloud compute tpus queued-resources list --help >/dev/null 2>&1; then \
		QCMD='gcloud --project=$(TPU_PROJECT) compute tpus queued-resources'; \
	else \
		QCMD='gcloud --project=$(TPU_PROJECT) alpha compute tpus queued-resources'; \
	fi; \
	for ZONE in $(TPU_QUEUE_ZONES); do \
		$$QCMD list --zone="$$ZONE" --format='value(name)' 2>/dev/null | while read -r NAME; do \
			case "$$NAME" in \
				qr-*) echo "Deleting $$NAME ($$ZONE)"; $$QCMD delete "$$NAME" --zone="$$ZONE" --quiet ;; \
			esac; \
		done; \
	done

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


.PHONY: pdf clean watch run.webapp test count-lines all
pdf: pdf.build
clean: pdf.clean
watch: pdf.watch
run.webapp: web.dev
test: test.backend
count-lines: stats.lines
all: pdf.build
