LATEXMK   := latexmk
SRCDIR    := paper/src
BUILDDIR  := build
TEX       := main.tex
JOBNAME   := main
PDF       := paper/$(BUILDDIR)/$(JOBNAME).pdf

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


.PHONY: all pdf clean watch run.webapp
