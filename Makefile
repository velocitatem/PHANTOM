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

.PHONY: help
help:
	@echo "pdf.build pdf.watch pdf.clean | test.backend test.e2e test.all | web.dev | install | stats.lines"

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
