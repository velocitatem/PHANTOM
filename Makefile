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

.PHONY: all pdf clean watch run.webapp install test
