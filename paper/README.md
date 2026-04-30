# paper/ — TMLR submission draft

Draft of the Bayesian X-Learner paper, targeting Transactions on
Machine Learning Research (TMLR).

## Layout

```
paper/
  main.tex              Root document; \input's the section files.
  tmlr.sty              Official TMLR style (JmlrOrg/tmlr-style-file).
  refs.bib              Bibliography.
  sections/             Main-body sections (target ~9 pages).
    00_abstract.tex
    01_introduction.tex
    02_background.tex
    03_related_work.tex
    04_method.tex
    05_experiments.tex
    06_discussion.tex
    07_conclusion.tex
  appendix/             Extended material (room to ~30-35 pages).
    A_figures.tex
    B_experiments_extended.tex
    C_evt_path.tex
    D_discussion_extended.tex
  figures/              (currently empty; populate from
                        benchmarks/results/figures/ when drafting.)
```

## Page targets (soft — TMLR has no hard limit)

| Section | Target pages |
|---|---:|
| Abstract + Introduction | 1.0 |
| Background & Landscape  | 1.0 |
| Related Work            | 1.0--1.5 |
| Method                  | 2.0 |
| Experiments             | 2.5 |
| Discussion              | 1.0 |
| Conclusion              | 0.3 |
| **Main-body total**     | **~9** |
| Appendix A (figures)    | 5--8 |
| Appendix B (experiments)| 12--15 |
| Appendix C (EVT path)   | 2--3 |
| Appendix D (discussion) | 3--5 |
| **Full total**          | **~30--40** |

## Build

### Overleaf (recommended)

1. Zip everything in `paper/` (see `paper.zip` at repo root, regenerated
   by `make paper-zip` or the one-liner below).
2. Overleaf → **New Project → Upload Project** → select the zip.
3. In Overleaf: **Menu → Compiler → pdfLaTeX**, **Main document →
   `main.tex`**.
4. Compile. BibTeX runs automatically on first compile.

Regenerate the upload zip from a shell:
```bash
cd paper && zip -r ../paper.zip . -x '*.aux' '*.log' '*.bbl' '*.blg' \
    '*.out' '*.toc' '*.synctex.gz' 'main.pdf' '.gitignore'
```

### Local build (if TeXLive is installed)

```bash
cd paper
latexmk -pdf main           # preferred
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Drafting status

- [x] Outline agreed (Section 3.2 reframed to
      "robust statistics and heavy-tailed inference for CATE").
- [x] Tail-heterogeneous CATE probe run; results in
      `benchmarks/results/tail_heterogeneous_cate.md`.
- [x] Skeleton with `\TODO{...}` markers committed.
- [ ] Section 1--7 prose drafts.
- [ ] Figures (Matplotlib/PGF) regenerated for paper quality.
- [ ] Bibliography completeness check.
- [ ] Anonymous vs signed submission decision.

Each `\TODO{...}` macro highlights a place that needs drafting; remove
the macro definition in `main.tex` before submission.

## Data sources

All empirical tables in the main body are drawn from:

- `benchmarks/results/ihdp_benchmark.md` — Table \ref{tab:ihdp}
- `benchmarks/results/whale_density_catboost_huber.md` — Table \ref{tab:whale}
- `benchmarks/results/tail_heterogeneous_cate.md` — Table \ref{tab:tail_signal}
- `benchmarks/results/huber_delta_followups.md` — cross-references

Raw CSVs alongside each markdown file.

## Reproducibility

A reproduction script exists at `scripts/reproduce.sh` (project root).
The appendix will cite the exact commands used for each table.
