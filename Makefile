PYTHONPATH=
SHELL=bash
VENV=../.venv
PROJECT=pyproject.toml

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

../.venv:
	cd ..
	make -s .venv

.PHONY: plot_reconstructions
plot_reconstructions: ../.venv
	$(VENV_BIN)/python scripts/plotting/plot_reconstruction.py \
		--channel_type Pospischil \
		--vae_type LinearVAE \
		--trace_index 0 \
		--model_date 2026_06_04_18_50_43_568540 -tm

# The region-of-success figure set. plot_region_optimizers.py runs FIRST: it
# owns the choice of arm (and writes plots/opt_region_scores/best_optimizer.json)
# that every camp below is drawn under.
.PHONY: plot_region_scores
plot_region_scores: ../.venv
	$(VENV_BIN)/python scripts/plotting/plot_region_optimizers.py
	$(VENV_BIN)/python scripts/plotting/plot_region_camps.py
	$(VENV_BIN)/python scripts/plotting/plot_region_cross.py
	$(VENV_BIN)/python scripts/plotting/plot_region_diagnostics.py
	$(VENV_BIN)/python scripts/plotting/plot_region_ladder.py

# Convexity descriptors vs the region-of-success score. CPU-only and about a
# minute -- every number comes off the two JSONL stores, nothing is simulated.
# JAX_PLATFORMS=cpu because the scripts import gen_region_success_scores (for
# `ladder_score`), which pulls in jax/jaxley and would otherwise claim a GPU.
.PHONY: plot_convexity_vs_opt
plot_convexity_vs_opt: ../.venv
	JAX_PLATFORMS=cpu $(VENV_BIN)/python scripts/plotting/plot_convexity_vs_opt_grid.py --per_set
	JAX_PLATFORMS=cpu $(VENV_BIN)/python scripts/plotting/plot_convexity_vs_opt_lines.py --stat all
	JAX_PLATFORMS=cpu $(VENV_BIN)/python scripts/plotting/plot_convexity_vs_opt_scatter.py --stat all --unit both
	JAX_PLATFORMS=cpu $(VENV_BIN)/python scripts/plotting/plot_convexity_vs_opt_scatter.py --stat all --unit seed --raw
	JAX_PLATFORMS=cpu $(VENV_BIN)/python scripts/plotting/plot_convexity_vs_opt_scatter.py --stat basin_wide --unit set

# The CPU half of the convexity pipeline: turn the cached descriptors.json files
# under data/loss_sweeps/ into data/convexity-scores.jsonl. Pure JSON reading --
# seconds, one process, no GPU -- which is why it is a target and not a study.
# The GPU half is configs/studies/convexity_sweeps.yaml.
.PHONY: collect_convexity_scores
collect_convexity_scores: ../.venv
	JAX_PLATFORMS=cpu $(VENV_BIN)/python scripts/datagen/collect_convexity_scores.py
