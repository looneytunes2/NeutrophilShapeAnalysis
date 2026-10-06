# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

Work only inside this repository (`C:\Users\Aaron\NeutrophilShapeAnalysis`). Do not edit files outside it without explicit permission. Raw image data lives on network shares / `E:/Aaron/...` (see `[data.*]` in `config.toml`); never modify or delete it.

## Environment and commands

- Windows, conda env `nsa` (Python 3.12). Install: `conda-lock install -n nsa conda-lock.yml`, then `pip install -e .` (package `neutrophil_shape`). `environment.yaml` is the exported spec; the package list for it is the `[dependencies]` table at the bottom of `neutrophil_shape/config/config.toml`.
- There is no test suite, linter, or build step. Code is validated by running notebooks/scripts and inspecting output.
- Figure scripts are run directly: `python figures/fig4/fig4_random_confocal_PC1-PC2_CGPS.py`.

## Pipeline (notebooks in `Notebooks/`, run in order)

1. `Segment_and_Track_Motility_Paper_{Confocal_Data,LLS_Random_Only}` — segment, track, crop single cells from large images.
2. `Processing_Motility_Paper_{Confocal_Data,LLS_Random_Only}` — compute trajectory/shape alignment angles, align cell meshes, compute spherical harmonic (SH) coefficients.
3. `PCA_with_all_37C_confocal_data` / `PCA_with_LLS_apply_confocal` — QC, PCA on SH coefficients of all confocal data; the same PCA transform is applied to LLS data.
4. `Processing_Motility_Paper_{Confocal,LLS}_Detailed_Balance` — coarse-grained phase spaces (CGPSs) over PC pairs, area enclosing rates (AER) for real and bootstrapped trajectories.

Figure/animation scripts in `figures/figN/`, `figures/figsN/` (supplement), `figures/animations/` and `Visualizations/` consume the outputs of step 3–4. `script_notebook/` is a dated scratch/exploration log (not library code, not imported), and `old_figures/` is ignored legacy.

## Architecture

**Config (`neutrophil_shape/config/`)** is the central hub. `load_config(microscope_type='confocal'|'lls')` reads `config.toml` into dataclasses (`models.py`). Every script then selects an alignment, which is what actually parameterizes the run:

```python
config = load_config(microscope_type='confocal')
config._alignment = 'trajectory'   # 'shape' | 'trajectory_shape' | 'trajectory'
```

The `_alignment` setter is a property with side effects: it sets `common.savedir` (`data/<alignment>_<microscope>/`, created on disk), the per-alignment PC flips and symmetrical PCs (and `pc_combos_sym`), and the CGPS `origins` for each PC pair. Forgetting to set it leaves these as `None`. `pc_combos` is all pairs of the top `npcs` PCs; `origins` is indexed in the same order as `pc_combos`. Thresholds for AER states are in `db_params.<microscope>.cycle_thresh` keyed by `"PCa-PCb"`. Confocal and LLS use different pixel sizes, bin counts, and thresholds, so always load the matching microscope config.

**Three alignments** are first-class and all outputs are duplicated per alignment under `data/{shape,trajectory_shape,trajectory}_{confocal,lls}/`. `data/` is git-ignored; typical contents are `shape_data/All_Data_with_CGPS_bins.csv` and `detailed_balance/`. Alignment-dependent code must not hardcode one alignment.

**`neutrophil_shape/CustomFunctions/`** (flat module collection, imported like `from neutrophil_shape.CustomFunctions import utils`):
- `image_processing.py`, `segment_cells2short.py`, `segment_LLS.py`, `track_functions.py`, `TrackMate_Script.py` — segmentation, tracking, cropping, alignment angle extraction.
- `shparam_mod.py`, `shtools_mod.py`, `cytoparam_mod.py` — modified forks of aicsshparam / pyshtools / cytoparam: mesh/image principal axes, SH coefficients, shape metrics, protrusion info, reconstructions. Principal axes for the confocal path are computed from image-volume coordinates (not surface-mesh coordinates); keep that consistent when editing.
- `DetailedBalance.py` — transition counting, trajectory interpolation, bootstrapping (graph with dictionary lookups; bootstrapped trajectories are reference lookups into raw data rather than copies), and AER computation. Heavy and performance-sensitive; uses multiprocessing (`*_wrapper` / `*_imap` functions exist so work can be mapped over arguments tuples).
- `utils.py` — trajectory smoothing, PC distances, AER smoothing/state classification and chunking, rate fits, image alignment helpers.
- `linear_cycle_utils.py`, `PCvisualization.py`, `shapePCAtools.py` — linearized CGPS cycles, shape-space visualization, PCA helpers.
- `file_management.py`, `stereotypyAvL.py` — mostly one-off scripts with module-level side effects (hardcoded paths); do not import them.

## Conventions

- Scripts are plain `.py` with top-level code (no `main`), usually built as: load config, set alignment, read CSVs from `config.common.savedir`, plot, save. Figure scripts are named `fig<N>_<description>.py`; PC pairs appear in names as `1-2`, `4-5`, `2-8`, `1-7`.
- PC indices in the config and in code are 1-based; negative entries in `pc_combos_sym` mean a symmetrical PC.
- Cell identity is `(Treatment, cell, CellID, frame)`; treatments include Random, galvanotaxis (Galv), CK666, and para-nitroblebbistatin (PNB).
- Notebooks are tracked in git; avoid editing them via text tools in ways that clobber outputs/metadata.
