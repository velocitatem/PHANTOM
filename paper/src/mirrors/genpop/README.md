# General Public Mirror

This directory contains a general-public edition of the PHANTOM thesis. Mathematical formulas and complex algorithms have been translated into plain language explanations while preserving the complete narrative and all research findings.

## Build Instructions

### Quick Build
From the project root:
```bash
make pdf.genpop
```

### Watch Mode (auto-rebuild on changes)
```bash
make pdf.genpop.watch
```

### Manual Build
```bash
cd paper/src
latexmk -pdf -jobname=main-genpop -f -interaction=nonstopmode -outdir=../build main-genpop.tex
```

## Output Location

The compiled PDF will be at:
```
paper/build/main-genpop.pdf
```

## What's Different?

### Original Technical Version
- Complex mathematical formulas and equations
- Formal algorithmic pseudocode
- Statistical notation and proofs
- Assumes advanced math background

### General Public Version
- Plain language explanations of formulas
- Step-by-step algorithm descriptions
- Intuitive statistical explanations
- Accessible to non-technical readers

## Structure

All mirrored chapters follow the same structure as the original:

- `01-intro.tex` - Introduction
- `02-literature-review.tex` - Literature Review
- `03-methodology.tex` - Methodology (most heavily adapted)
- `04-results.tex` - Results
- `05-discussion.tex` - Discussion
- `06-conclusion.tex` - Conclusion

## Translation Approach

Following the instructions in `INSTRUCTIONS.md`, we:

1. **Preserve** all language and phrasing from the original
2. **Replace** mathematical formulas with inline plain-language explanations
3. **Simplify** complex algorithms into readable step descriptions
4. **Maintain** all citations, figures, tables, and narrative flow
5. **Keep** technical terms when commonly understood
6. **Explain** technical concepts inline for general readers

## Example Transformations

**Mathematical formula:**
```latex
\hat{q}_{t,i} = \sum_{s \in \mathcal{S}_t} \sum_{k=1}^{L_s} \omega(a_{s,k}) \cdot \mathbb{1}[i_{s,k} = i]
```

**Becomes:**
"for each session in a time period, we sum up all the events where a specific product was interacted with, and we weight those events by how strong a signal they provide about willingness to pay"

**Proof notation:**
```latex
P(p_{(1)} > t) = [1 - F(t)]^N \to 0
```

**Becomes:**
"When we raise a number less than 1 to higher and higher powers (as N grows), it decays exponentially toward zero"
