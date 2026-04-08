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

# Biber runs with cwd paper/build; \addbibresource{bib/references.bib} must resolve there.
# Symlink makes biber log 'bib/references.bib' (not ../src/...) so latexmk's post-check passes.
link_build_bib() {
  ln -sfn ../src/bib ../build/bib
}

case "$cmd" in
  build)
    mkdir -p paper/build
    sync_mdp_figures
    bash paper/concat_code.sh
    cd paper/src
    link_build_bib
    latexmk -pdf -jobname=main -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build main.tex
    ;;
  watch)
    mkdir -p paper/build
    sync_mdp_figures
    cd paper/src
    link_build_bib
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
    link_build_bib
    latexmk -pdf -jobname=main-genpop -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build main-genpop.tex
    ;;
  watch-genpop)
    mkdir -p paper/build
    sync_mdp_figures
    cd paper/src
    link_build_bib
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
  build-summary)
    mkdir -p paper/build
    cd paper/src
    link_build_bib
    latexmk -pdf -jobname=summary -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build summary.tex
    ;;
  watch-summary)
    mkdir -p paper/build
    cd paper/src
    link_build_bib
    latexmk -pvc -pdf -jobname=summary -f -interaction=nonstopmode -file-line-error -r ../.latexmkrc -outdir=../build summary.tex
    ;;
  *)
    printf '%s\n' "Unknown paper command: $cmd" >&2
    exit 1
    ;;
esac
