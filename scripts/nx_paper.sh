#!/usr/bin/env bash

set -euo pipefail

cmd="${1:-}"

case "$cmd" in
  build)
    mkdir -p paper/build
    bash paper/concat_code.sh
    cd paper/src
    latexmk -pdf -jobname=main -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build main.tex
    ;;
  watch)
    mkdir -p paper/build
    cd paper/src
    latexmk -pvc -pdf -jobname=main -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build main.tex
    ;;
  clean)
    cd paper/src
    latexmk -C -jobname=main -outdir=../build || true
    rm -rf ../build/*
    ;;
  wordcount)
    printf '%s\n' 'Counting words in main text (excluding appendix)...'
    texcount -nosub -total -sum -1 \
      paper/src/chapters/01-intro.tex \
      paper/src/chapters/02-literature-review.tex \
      paper/src/chapters/03-methodology.tex \
      paper/src/chapters/04-results.tex \
      paper/src/chapters/05-discussion.tex \
      paper/src/chapters/06-conclusion.tex
    ;;
  build-genpop)
    mkdir -p paper/build
    cd paper/src
    latexmk -pdf -jobname=main-genpop -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build main-genpop.tex
    ;;
  watch-genpop)
    mkdir -p paper/build
    cd paper/src
    latexmk -pvc -pdf -jobname=main-genpop -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build main-genpop.tex
    ;;
  *)
    printf '%s\n' "Unknown paper command: $cmd" >&2
    exit 1
    ;;
esac
