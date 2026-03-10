The source thesis lives in paper/src/ (main.tex + chapters/ + preamble.tex + bib/references.bib).
Your job is to produce a self-contained arxiv-ready submission in this directory.

## What to produce

A single compilable main.tex that reproduces the full thesis content and compiles
cleanly with pdflatex + bibtex (no latexmk extras, no shell-escape). The output
must pass arxiv's AutoTeX pipeline without manual intervention.

## arxiv constraints to satisfy

1. Flat layout: all files referenced by \input or \includegraphics must sit in
   this directory or one level below (no ../../ relative paths). Copy or symlink
   chapters and graphics here, then rewrite \input paths accordingly.

2. Bibliography: arxiv does not run biber. Use bibtex. The preamble currently
   loads biblatex with backend=bibtex — keep that, or switch to natbib +
   \bibliographystyle{plainnat} if biblatex causes trouble. Either way, include
   the pre-built .bbl file in the submission tarball (arxiv runs bibtex once but
   having the .bbl avoids failures).

3. Packages: remove or replace anything arxiv's TeX Live snapshot may not carry.
   Known problematic ones in the current preamble:
   - newtxtext/newtxmath: fine on recent TeX Live, but have a fallback to
     \usepackage{times} + \usepackage{mathptmx} if the build fails.
   - algorithm2e: arxiv supports it, keep it.
   - cleveref: fine.
   - pgfplots: fine, but pin compat=1.18.

4. No \include of generated code appendix: the concat_code.sh appendix that gets
   appended at build time is for the university submission only. Omit it here or
   replace with a short note pointing to the repository URL.

5. Hyperref: keep it but add \usepackage[hidelinks]{hyperref} to suppress colored
   boxes, which look bad in arxiv's PDF renderer.

6. Title / author block: use a normal \maketitle with full author name, affiliation,
   and date. Do not use the titlepage environment from the university version.

7. Double spacing: remove \doublespacing (arxiv readers expect single spacing).

8. Page headers: remove the fancyhdr block; arxiv adds its own stamp header.

## Minimal diff principle

Preserve all content, section order, equations, theorems, figures, and tables from
the original thesis exactly. Only make the structural changes listed above. Do not
paraphrase, summarize, or rewrite prose. The mirror is a format adaptation, not
an editorial one.

## How to verify

From paper/src/mirrors/arxiv/ run:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

The build must complete without errors (warnings are acceptable). The resulting
main.pdf should be visually equivalent to paper/build/main.pdf modulo formatting
differences from removing double spacing and the titlepage.

To build via make from the repo root:
  make pdf.arxiv
