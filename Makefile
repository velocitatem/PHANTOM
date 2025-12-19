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
		-r ../.latexmkrc \
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

wordcount:
	@echo "Counting words in main text (excluding appendix)..."
	@texcount -nosub -total -sum -1 \
		$(SRCDIR)/chapters/01-intro.tex \
		$(SRCDIR)/chapters/02-literature-review.tex \
		$(SRCDIR)/chapters/03-methodology.tex \
		$(SRCDIR)/chapters/04-results.tex \
		$(SRCDIR)/chapters/05-discussion.tex \
		$(SRCDIR)/chapters/06-conclusion.tex

.PHONY: all pdf clean watch run.webapp install test wordcount
