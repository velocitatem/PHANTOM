LATEXMK   := latexmk
SRCDIR    := src
BUILDDIR  := build
TEX       := main.tex
JOBNAME   := main
PDF       := $(BUILDDIR)/$(JOBNAME).pdf

all: pdf

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

pdf: $(BUILDDIR)
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
	rm -rf $(BUILDDIR)/*


.PHONY: all pdf clean watch
