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

.DEFAULT_GOAL := help

all: pdf

run.webapp:
	@cd web && npm install && npm run dev

$(BUILDDIR):
	mkdir -p paper/$(BUILDDIR)

pdf: $(BUILDDIR)
	@echo "Concatenating source code..."
	@bash paper/concat_code.sh
	@cd $(SRCDIR) && \
	$(LATEXMK) -pdf -jobname=$(JOBNAME) \
		-interaction=nonstopmode -file-line-error \
		-outdir=../$(BUILDDIR) $(TEX)

watch: $(BUILDDIR)
	@cd $(SRCDIR) && \
	$(LATEXMK) -pvc -pdf -jobname=$(JOBNAME) \
		-interaction=nonstopmode -file-line-error \
		-r ../.latexmkrc \
		-outdir=../$(BUILDDIR) $(TEX)

clean:
	@cd $(SRCDIR) && \
	$(LATEXMK) -C -jobname=$(JOBNAME) -outdir=../$(BUILDDIR) || true
	rm -rf paper/$(BUILDDIR)/*

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)
	$(PIP) install -r requirements.txt

test: $(VENV)
	$(PYTEST) -v

count-lines:
	@find . \( -path '*/node_modules' -o -path '*/.venv' -o -path '*/venv' \) -prune -o \
	\( -name "*.ts" -o -name "*.py" \) -type f -print0 | xargs -0 cat | wc -l

test.e2e:
	@echo "Installing E2E dependencies..."
	@cd tests/e2e && npm install
	@cd tests/e2e && npx playwright install chromium --with-deps
	@echo "Waiting for services..."
	@timeout 30 bash -c 'until curl -sf http://localhost:5000/health > /dev/null 2>&1; do sleep 1; done' || (echo "Backend not ready" && exit 1)
	@timeout 30 bash -c 'until curl -sf http://localhost:3000 > /dev/null 2>&1; do sleep 1; done' || (echo "Web app not ready" && exit 1)
	@echo "Running E2E tests..."
	@cd tests/e2e && npm test

.PHONY: all pdf clean watch run.webapp install test test.e2e count-lines
