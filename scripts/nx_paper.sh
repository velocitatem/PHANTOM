#!/usr/bin/env bash

set -euo pipefail

cmd="${1:-}"

sync_mdp_figures() {
  local script_dir project_root sim_dir chapters_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  project_root="$(cd "$script_dir/.." && pwd)"
  sim_dir="$project_root/sim/rl/behavior_loader"
  chapters_dir="$project_root/paper/src/chapters"

  printf '%s\n' 'Refreshing MDP figures for paper...'
  (
    cd "$sim_dir"
    python models.py
  )

  cp "$sim_dir/human_mdp_viz.pdf" "$chapters_dir/mdp_human.pdf"
  cp "$sim_dir/agent_mdp_viz.pdf" "$chapters_dir/mdp_agent.pdf"
}

case "$cmd" in
  build)
    mkdir -p paper/build
    sync_mdp_figures
    bash paper/concat_code.sh
    cd paper/src
    latexmk -pdf -jobname=main -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build main.tex
    ;;
  watch)
    mkdir -p paper/build
    sync_mdp_figures
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
    sync_mdp_figures
    cd paper/src
    latexmk -pdf -jobname=main-genpop -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build main-genpop.tex
    ;;
  watch-genpop)
    mkdir -p paper/build
    sync_mdp_figures
    cd paper/src
    latexmk -pvc -pdf -jobname=main-genpop -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build main-genpop.tex
    ;;
  build-arxiv)
    mkdir -p paper/build
    cd paper/src/mirrors/arxiv
    pdflatex -interaction=nonstopmode -file-line-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -file-line-error main.tex
    pdflatex -interaction=nonstopmode -file-line-error main.tex
    cp main.pdf ../../../build/main-arxiv.pdf
    ;;
  *)
    printf '%s\n' "Unknown paper command: $cmd" >&2
    exit 1
    ;;
esac
