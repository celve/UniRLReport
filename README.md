# UniRL Paper

Manuscript: **UniRL: Trajectory-Centric Post-Training Across Heterogeneous
Generative Models**.

This is the paper/Overleaf repository, not the training-code repository.
Overleaf's root document is `main.tex`; keep the TMLR style, bibliography and
relative figure/table inputs together. `main.pdf` is the reviewed local build.
GPU results are intentionally pending, not inferred from source code or filled
with estimates.

## GPU handover

Start with [GPU_HANDOFF_README.md](GPU_HANDOFF_README.md). Its readiness table
distinguishes existing launchers from prerequisites that the GPU-side agent must
implement and validate before formal runs. Every experiment has settings,
commands, expected signals, failure ownership and a paper destination.

- [EXPERIMENT_RUNBOOK.md](EXPERIMENT_RUNBOOK.md): evidence-admission and statistical rules.
- [PAPER_REVIEW.md](PAPER_REVIEW.md): substantive review, paragraph roles and remaining scientific obligations.
- [PAPER_PLAN.md](PAPER_PLAN.md): thesis, source audit, claim-to-evidence map and priorities.
- [experiments/trajectory_ir/README.md](experiments/trajectory_ir/README.md): historical CPU-only IR evidence and reproduction instructions.

Train and implement missing GPU harnesses in the separately pinned UniRL code
checkout. Store checkpoints, datasets, environments and raw GPU logs outside this
repository. Return compact, artifact-generated tables/figures and their manifests
here. Do not upload model weights, credentials, private prompts or large logs to
GitHub/Overleaf.

## Build

Compile `main.tex` in Overleaf, or with an installed LaTeX toolchain:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Keep unknown hardware, dataset statistics and final scores blank until supported
by admitted artifacts. Compile and inspect the PDF after changing the manuscript.
