#!/usr/bin/env bash
# Render every figure in figures/tikz/ to output/figures/<name>.pdf.
# Run from the repository root, after the analysis scripts:  bash figures/build_figures.sh
# Needs pdflatex with TikZ and pgfplots.
set -e
cd "$(dirname "$0")/.."
TMP=output/figures/tmp
mkdir -p "$TMP"
for src in figures/tikz/*.tex; do
  name=$(basename "$src" .tex)
  [ "$name" = styles ] && continue
  cat > "$TMP/_wrap.tex" <<TEX
\documentclass[border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,shapes.geometric,decorations.pathreplacing}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\input{figures/tikz/styles}
\begin{document}
\input{$src}
\end{document}
TEX
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$TMP" -jobname="$name" "$TMP/_wrap.tex" >/dev/null
  mv "$TMP/$name.pdf" "output/figures/$name.pdf"
  echo "built output/figures/$name.pdf"
done
rm -rf "$TMP"
