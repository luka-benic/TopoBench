# Training MACE and TopoBench on OpenQDC datasets

This folder trains two kinds of models on the **energies** of molecules from [OpenQDC](https://github.com/valence-labs/OpenQDC) datasets, through one shared data pipeline:

- **TopoBench** models (graph, simplicial and cell neural networks) predict one energy per molecule.
- **MACE** (an equivariant interatomic potential) learns the total energy of each molecule from atomic positions.

Both pipelines read the data through the same loader, so they see the same molecules, in the same units, with the same energy column. Training is **energy-only**: forces are never loaded.

```
                       OpenQDC data (downloaded once, cache folder)
                                      |
                          OpenQDCSource  (shared loader, eV and Angstrom)
                         /                                  \
        OpenQDCDataset (PyG graphs)                 export_xyz.py (.xyz files)
                |                                              |
   python -m topobench (Hydra, any model)                mace_run_train
                |                                              |
        train_topobench.py                              train_mace.py
                         \                                  /
                          sub_openqdc.sh  (GPU or CPU, one command)
```

## Contents

1. [Quick start](#quick-start)
2. [What is where](#what-is-where)
3. [Concepts and conventions](#concepts-and-conventions)
4. [The datasets](#the-datasets)
5. [Choosing the energy column](#choosing-the-energy-column)
6. [Dataset yaml reference](#dataset-yaml-reference)
7. [Downloading and preprocessing](#downloading-and-preprocessing)
8. [Splits (identical for MACE and TopoBench)](#splits-identical-for-mace-and-topobench)
9. [Training TopoBench](#training-topobench)
10. [Training MACE](#training-mace)
11. [Choosing the metric and the loss](#choosing-the-metric-and-the-loss)
12. [Test-set results: .txt files and parity plots](#test-set-results-txt-files-and-parity-plots)
13. [Weights & Biases logging](#weights--biases-logging)
14. [Training on entire datasets](#training-on-entire-datasets)
15. [The submission script (GPU or CPU)](#the-submission-script-gpu-or-cpu)
16. [Running several jobs at once](#running-several-jobs-at-once)
17. [Where files are saved](#where-files-are-saved)
18. [Generating dataset configs](#generating-dataset-configs)
19. [Using the data in Python](#using-the-data-in-python)
20. [Viewing molecules, edges and liftings](#viewing-molecules-edges-and-liftings)
21. [Visualising MACE's cutoff radius and number of layers](#visualising-maces-cutoff-radius-and-number-of-layers)
22. [Comparing MACE and TopoBench](#comparing-mace-and-topobench)
23. [Troubleshooting](#troubleshooting)
24. [Known limitations](#known-limitations)
25. [Development: tests and checks](#development-tests-and-checks)

## Quick start

Run everything from the repository root, with the project's uv environment (`.venv`).

```bash
cd /gpustorage2/lukab/TopoBench
source .venv/bin/activate          # or call .venv/bin/python directly
nvidia-smi                          # pick a GPU with free memory

# TopoBench (a graph model) on GPU 3, QM9, energy column 0
scripts/openqdc/sub_openqdc.sh topobench 3 --dataset QM9 --model graph/gcn --max-samples 20000

# MACE on GPU 3
scripts/openqdc/sub_openqdc.sh mace 3 --dataset QM9 --max-samples 20000 --epochs 50

# the same on the CPU
scripts/openqdc/sub_openqdc.sh topobench cpu --dataset QM9 --max-samples 2000 --epochs 5
```

The first run downloads the dataset and preprocesses it; later runs reuse both. Read [Known limitations](#known-limitations) before using QM9 or QM7b energies: OpenQDC's own unit label is wrong for them.

## What is where

| Path | What it is |
|---|---|
| `scripts/openqdc/sub_openqdc.sh` | One command to train MACE or TopoBench on a chosen GPU or on the CPU |
| `scripts/openqdc/train_topobench.py` | Prepares the data, then runs `python -m topobench` on an OpenQDC dataset |
| `scripts/openqdc/train_mace.py` | Prepares the `.xyz` files, then runs `mace_run_train` |
| `scripts/openqdc/prepare_data.py` | Download and preprocess only (also called by both training scripts) |
| `scripts/openqdc/export_xyz.py` | The `.xyz` export used by MACE, also usable on its own ([options](#exporting-the-xyz-files-on-its-own)) |
| `scripts/openqdc/generate_configs.py` | Writes the dataset yaml files |
| `configs/dataset/graph/openqdc_<name>.yaml` | One config per dataset (7 in the repo) |
| `topobench/data/utils/openqdc_utils.py` | Shared loader (`OpenQDCSource`), unit constants, integrity check, locks |
| `topobench/data/datasets/openqdc_dataset.py` | `OpenQDCDataset`: OpenQDC molecules as PyG graphs |
| `topobench/data/loaders/graph/openqdc_datasets_loader.py` | `OpenQDCDatasetLoader`, found automatically by TopoBench |
| `topobench/data/utils/split_utils.py` | `random_splitting`, made safe for concurrent jobs (not used by the OpenQDC configs any more) |
| `topobench/data/utils/openqdc_utils.py` (`split_molecules`, `split_positions`, `molecular_edges`) | the train/valid/test split shared by TopoBench and MACE, and the edge rule shared by the dataset and the viewer |
| `topobench/utils/openqdc_viewer.py`, `openqdc_liftings.py` | Interactive 3D viewer of molecules with their bond and cutoff edges and the liftings of TopoBench models |
| `topobench/utils/mace_viewer.py` | Interactive view of MACE's cutoff radius and number of layers (receptive field of an atom), from the geometry alone |
| `topobench/utils/openqdc_results.py` | Metrics, `.txt` writers, `save_split` and the parity plot (shared by both pipelines) |
| `topobench/callbacks/openqdc_test_results.py` | TopoBench callback that saves `split.txt` when training starts and the test-set predictions when it ends |
| `scripts/openqdc/mace_test_results.py` | Evaluates the best MACE model on `test.xyz` and saves the same files; `save_mace_split` writes `split.txt` |
| `scripts/openqdc/run_mace_train.py` | Starts MACE's training for energy-only data: skips the force terms of the loss so forces need not be computed, and makes W&B logging work without forces |
| `test/data/utils/test_openqdc_*.py`, `test_split_concurrency.py`, `test/utils/test_openqdc_{results,export,mace_launcher,viewer}.py`, `test/utils/test_mace_viewer.py` | Unit tests |

## Concepts and conventions

### Units

Every dataset is loaded with OpenQDC's own conversion: `energy_unit="ev"` and `distance_unit="ang"`. The values OpenQDC returns are used **as they are**. The raw files in the cache are never modified (OpenQDC converts in memory).

Forces are not read, so there is no force unit in this pipeline.

### Several energies per molecule

A dataset stores **one energy per quantum-chemistry method** for each molecule. The columns of the `energies` array are called *energy columns* (in the code: `energy_method`). QM9 has 9 columns, QM7b has 76, MultixcQM9 has 229, Alchemy has 1. A training run uses exactly one column. The list of columns (index and name) is at the top of each dataset yaml. See [Choosing the energy column](#choosing-the-energy-column).

### Target

`target` selects which energy of the chosen column is learned:

| `target` | Meaning |
|---|---|
| `energies` (default) | the total energy of the molecule |
| `formation_energies` | total energy minus the sum of OpenQDC's isolated-atom reference energies |
| `per_atom_formation_energies` | formation energy divided by the number of atoms (TopoBench only). TopoBench's losses and metrics are then per atom, like MACE's, and its readout averages over the atoms (`train_topobench.py` sets `model.readout.pooling_type=mean`, because the default sum pooling would make the network learn the 1/N factor); the saved test-set results are converted back to energies per molecule |

Both pipelines use the **total energy** by default. For datasets without isolated-atom reference energies (QM9, QM8, QM7b) the formation energy equals the total energy, so the choice only matters for the other datasets (for Alchemy, total energies are about -13,000 eV and formation energies about -100 eV). Total energies are large numbers, which is why both pipelines subtract a per-element baseline (see below).

### What the TopoBench graph contains

`OpenQDCDataset` turns each molecule into one PyG graph:

| Field | Content |
|---|---|
| `x` | one-hot atomic number over `atom_types`, plus the formal atomic charge (so `len(atom_types) + 1` features) |
| `pos` | coordinates in Angstrom |
| `z` | atomic numbers |
| `edge_index` | bonds inferred from distances: atoms are bonded if `d < bond_scale * (r_i + r_j)` with covalent radii (`edge_method: bonds`), or all pairs within `cutoff` (`edge_method: radius`) |
| `openqdc_idx`, `n_atoms` | the OpenQDC molecule index and the number of atoms (identify the molecules in the test-set results) |
| `energy` | the selected energy in eV, exactly as OpenQDC returns it (float64) |
| `y` | the regression target: `energy` minus a per-element baseline (float32) |

### The per-element baseline

Molecular energies are dominated by the energy of the atoms. With `baseline: per_element` (default) TopoBench fits one energy per element by least squares and trains on the remainder `y = energy - sum(count_z * e0_z)`. The fitted values are written to `baseline.json` next to the graphs. The fit uses the **training split only**, exactly like MACE's `--E0s=average` (the same least-squares problem on the same molecules), so both pipelines train on the same quantity, the energy minus the same per-element baseline. This was checked on Alchemy (2000 molecules): the fitted values in `baseline.json` equal MACE's printed `Atomic Energies used` to every digit. MACE's `std_scaling` is a normalisation inside the network and does not change the target. `baseline: none` trains on the raw energy. The default is `none` for datasets whose energies are not total energies (QM8). This equality with MACE holds for the targets `energies` and `formation_energies`; for `per_atom_formation_energies` TopoBench regresses on atom fractions instead of counts (an intensive target), and `baseline.json` then holds per-atom values.

The TopoBench graph folder is `datasets/graph/openqdc/<Dataset>/<tag>/` with the tag `<target>_m<column>_z<element hash>_b<baseline>[frac]_<bonds1.3 or radius5.0>_n<max_samples or all>_s<subsample seed>_split_v<valid>_t<test>_s<split seed>`, for example `energies_m0_zab12cd_bper_element_bonds1.3_nall_s0_split_v0.1_t0.1_s0`. The column is always written as an index (`3`, `"3"` and its name share one folder), the element hash makes a change of `atom_types` (the width of the node features) use a new folder, and, because the baseline depends on the split, the split settings are part of it: a different split seed converts the graphs again. `baseline.json` and `processed/data.pt` are written to a temporary file first and renamed, so a killed job leaves no truncated cache. All TopoBench models (graph, hypergraph, simplicial, cell) read the same `y` from these graphs, so they all train on the same energies as MACE.

### What is and is not changed relative to the raw data

| | |
|---|---|
| Raw files in the cache | never modified |
| Energies and positions | exactly as OpenQDC returns them for eV and Angstrom |
| Molecules with no finite energy in the chosen column | skipped |
| TopoBench target `y` | baseline-subtracted by default (the original energy is kept as `energy`) |
| Edges | built from distances (OpenQDC has no bonds) |

## The datasets

Seven datasets have a config in the repo. Any other OpenQDC potential-energy dataset can be added (see [Generating dataset configs](#generating-dataset-configs)).

| Dataset | Molecules | Energy columns | Elements | Notes |
|---|---|---|---|---|
| QM9 | 133,885 | 9 (column 0 = U0) | H C N O F | OpenQDC labels the energies hartree but they are kcal/mol (see below). Tested with TopoBench and MACE |
| MultixcQM9 | 133,631 | 229 | H C N O F | 229 functional/basis-set columns. The values are atomization-type energies (about -4 to -7 eV per atom), not total energies. Tested with TopoBench |
| QM7 | - | 80 | - | Config is a placeholder: the remote OpenQDC copy contains only zeros, training stops with an error |
| QM7b | 7,211 | 76 | H C N O S Cl | Same label problem as QM9 |
| Alchemy | 202,579 | 1 | H C N O F S Cl | Labels look right. Tested with TopoBench and MACE |
| VQM24 | 1,104,982 | 1 | 10 elements | Large: start with `--max-samples` |
| TMQM | 86,665 | 1 | 44 elements incl. transition metals | Up to 569 atoms |

The column counts and sizes are read from the generated yaml files. "Tested" means a short run went through; no run here was a benchmark.

## Choosing the energy column

The column is the `energy_method` entry of the dataset yaml. Index and name are interchangeable.

| How | Example |
|---|---|
| in the dataset yaml | `energy_method: 3` in `configs/dataset/graph/openqdc_qm9.yaml` |
| flag of the training scripts | `--energy-method 3` or `--energy-method "HF:sto-3g"` |
| environment variable of `sub_openqdc.sh` | `ENERGY_METHOD=3 sub_openqdc.sh ...` |
| Hydra override (direct `python -m topobench`) | `dataset.loader.parameters.energy_method=3` |
| write it into the yaml | `generate_configs.py --datasets QM9 --energy-method 3` |

Rules:

- **Indices start at 0.** QM9 has 0 to 8, QM7b 0 to 75, MultixcQM9 0 to 228; one-column datasets only have 0.
- Without a flag, both pipelines use the yaml's value. A flag or variable overrides it **for that run only**; the yaml file is not changed. (The exception: whenever `train_topobench.py` or `prepare_data.py topobench` has to write the yaml, because it is missing, is a `NOT DOWNLOADED` placeholder, or `--regen-config` is given, an explicit column is written into it.)
- If both `ENERGY_METHOD` and `--energy-method` are given, the flag wins.
- Names that contain spaces or colons need quotes.
- An index out of range stops the run with an error that lists the first column names.
- The chosen column is printed when a run starts, for example `Energy column 3: 'HF:sto-3g'`.

QM9 columns, for reference:

| Index | Name |
|---|---|
| 0 | Internal energy at 0 K (U0, the standard QM9 target) |
| 1 | B3LYP:def2-svp |
| 2 | HF:cc-pvtz |
| 3 | HF:sto-3g |
| 4 | PBE:sto-3g |
| 5 | SVWN:sto-3g |
| 6 | WB97X-D:aug-cc-pvtz |
| 7 | WB97X-D:def2-svp |
| 8 | WB97X-D:def2-tzvp |

Energies of different columns are on different scales (several Hartree apart); never mix columns in one training.

## Dataset yaml reference

`configs/dataset/graph/openqdc_<name>.yaml` is generated by `generate_configs.py`. Its header lists the number of molecules, atoms and all energy columns. Parameters under `loader.parameters`:

| Parameter | Default | Meaning |
|---|---|---|
| `data_name` | dataset name | OpenQDC class name |
| `openqdc_cache_dir` | `$OPENQDC_CACHE_DIR`, else `<repo>/datasets/openqdc` | Where OpenQDC data is stored |
| `data_dir` | `<repo>/datasets/graph/openqdc` | Where the converted graphs are stored |
| `atom_types` | detected | Elements used for the one-hot node features |
| `target` | `energies` | See [Target](#target) |
| `energy_method` | `0` | The energy column, see above |
| `baseline` | `per_element` (`none` for QM8) | See [The per-element baseline](#the-per-element-baseline) |
| `edge_method` | `bonds` | `bonds` or `radius` |
| `bond_scale` | `1.3` | Bond if `d < bond_scale * (r_i + r_j)` |
| `cutoff` | `5.0` | Angstrom, used when `edge_method: radius` |
| `max_samples` | `null` (all) | Random subset of molecules |
| `subsample_seed` | `0` | Seed of that subset |
| `valid_fraction`, `test_fraction`, `split_seed` | `0.1`, `0.1`, `0` | Fractions and seed of the shared split, see [Splits](#splits-identical-for-mace-and-topobench) |
| `data_domain`, `data_type` | `graph`, `openqdc` | Fixed; they build the `data_dir` path |

Other sections: `parameters` (regression task, `mse` loss, `mae` as monitored metric, `num_features`), `split_params` (`split_type: fixed`: the split comes from the dataset, see [Splits](#splits-identical-for-mace-and-topobench)) and `dataloader_params` (batch size 128). Every parameter can be overridden on the command line with `dataset.loader.parameters.<name>=<value>`.

MACE reads `target`, `max_samples`, `subsample_seed`, the split settings and `energy_method` from the same yaml when the flag is not given (a yaml `target` that MACE cannot learn, `per_atom_formation_energies`, stops it with a message). Regenerating the yaml (`--regen-config`) resets hand-edited loader parameters except the energy column.

The `atom_types` list is detected from the data. If a config was written before the data was downloaded it carries a generic element list and a `NOT DOWNLOADED` note; the training script then downloads the dataset and rewrites the file.

## Downloading and preprocessing

Every training run starts with a check and does only what is missing, in this order:

1. For TopoBench only: the **dataset yaml** exists (otherwise it is written, which downloads the data to detect the elements). MACE runs never write the yaml; they only read it if it exists (else the defaults: column 0, split 0.1/0.1/seed 0).
2. The **OpenQDC data** is downloaded and complete.
3. The **preprocessed data** exists for exactly these settings: the TopoBench graphs (for TopoBench) or the `.xyz` files (for MACE). Otherwise they are built.

Progress is printed as `[prepare] ...` lines. The same step can be run alone, for example once before starting many jobs:

```bash
python scripts/openqdc/prepare_data.py topobench --dataset QM9 --energy-method 3 --max-samples 20000
python scripts/openqdc/prepare_data.py topobench --dataset QM9 --loader-param baseline=none
python scripts/openqdc/prepare_data.py mace --dataset QM9 --max-samples 20000
```

`--loader-param KEY=VALUE` (repeatable) sets any other loader parameter of the yaml. `prepare_data.py mace` takes the same data flags as `train_mace.py`; `prepare_data.py topobench` also accepts `--split-seed`, `--valid-fraction` and `--test-fraction`.

**Interrupted downloads.** OpenQDC writes its files in place and trusts any file that exists, so a download killed midway used to leave a truncated file and a confusing error. Before every load the files are now checked against each other (sizes, number of atoms, molecules and energies, and that the properties file is complete). If a download was cut off, the partial files are removed (`[repair] ...`) and downloaded again.

**Messages you may see**

| Message | Meaning |
|---|---|
| `[prepare] ...` | progress of the check |
| `[wait] ... another job is busy with it, waiting ...` | another job is downloading or preprocessing the same data; this job continues by itself afterwards |
| `[repair] ... incomplete files found` | a previous download was interrupted; the dataset is downloaded again |

Model-specific preprocessing (the simplicial and cell lifts) still happens during training; TopoBench protects it with its own lock.

## Splits (identical for MACE and TopoBench)

Both pipelines use one function, `split_molecules` in `topobench/data/utils/openqdc_utils.py`. It takes the **OpenQDC index** of every molecule, sorts the indices, permutes them with `numpy.random.default_rng(split_seed)` and cuts them into validation (first `valid_fraction`), test (next `test_fraction`) and training (rest). The result depends only on the set of molecules, the fractions and the seed, not on the order in which a pipeline lists them, so no split files have to be shared.

| | TopoBench | MACE |
|---|---|---|
| Where | `OpenQDCDataset.split_idx` (`split_type: fixed` in the yaml) | `partition_frames` in `export_xyz.py` |
| Parameters | `dataset.loader.parameters.{valid_fraction,test_fraction,split_seed}` or `--valid-fraction --test-fraction --split-seed` (default from the dataset yaml: 0.1, 0.1, 0) | the same flags; the default is read from the same yaml |

The split settings are part of both folder names (TopoBench graphs: `..._split_v0.1_t0.1_s0`, MACE `.xyz` files: `split_v0.1_t0.1_s0`), because the per-element baseline is fitted on the training split. A new `--split-seed` therefore converts or exports the data again. The baseline is the same for both pipelines, see [The per-element baseline](#the-per-element-baseline).

The split is identical when both runs use the same **dataset, energy column, target, `--max-samples`, `--subsample-seed`** (they decide which molecules exist; molecules without an energy for the column are dropped by both) and split settings. This was checked on Alchemy (2000 molecules): the train, validation and test index sets of the `.xyz` files equal those of TopoBench, and both `test_predictions.txt` files list the same 200 molecules.

**Every run saves its own split.** Before training starts, each run writes `test_results/split.txt` in its own run folder (TopoBench: `logs/train/runs/<timestamp>_<pid>/`, MACE: `logs/mace_openqdc/<run name>/`). It has one line per molecule, `openqdc_idx` and `train`, `valid` or `test`, plus a `# counts:` header, and loads with `numpy.loadtxt(path, dtype=str)`. The file is written even if the run later stops (also with `--no-test-results`), and runs never share it. Two runs of the two pipelines on the same settings produce identical files (checked on Alchemy).

For several independent splits use different `--split-seed` values (for example 0 to 4) in both scripts. A split that keeps related molecules together (scaffold or conformer groups) would need its own function with the same interface.

## Training TopoBench

```bash
python scripts/openqdc/train_topobench.py --dataset QM9 --model graph/gcn --energy-method 0 \
    --max-samples 20000 --epochs 50
```

The script prepares the data (see above), then runs `python -m topobench` with the right overrides. Anything after a lone `--` is passed to Hydra unchanged (it comes after the flags, so it wins over them, and the data preparation uses the same values):

```bash
python scripts/openqdc/train_topobench.py --dataset QM9 --model graph/gcn -- \
    optimizer.parameters.lr=0.001 dataset.loader.parameters.edge_method=radius
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | required | OpenQDC dataset name, case-insensitive |
| `--model` | `graph/gcn` | Any model config under `configs/model/`, e.g. `graph/gin`, `simplicial/scn`, `cell/cwn`, `hypergraph/unignn2` |
| `--energy-method` | from the yaml | Energy column, index or name |
| `--target` | from the yaml (`energies`) | `energies`, `formation_energies` or `per_atom_formation_energies` |
| `--max-samples` | all | Random subset of molecules (recommended for datasets above ~500k molecules) |
| `--subsample-seed` | from the yaml | Seed of that subset |
| `--split-seed`, `--valid-fraction`, `--test-fraction` | from the yaml (0, 0.1, 0.1) | The shared split, see [Splits](#splits-identical-for-mace-and-topobench) |
| `--epochs` | from the config | `trainer.max_epochs` |
| `--batch-size` | 128 | Dataloader batch size |
| `--seed` | from the config | Global seed |
| `--logger` | `csv` | `csv`, `wandb`, `tensorboard`, ... |
| `--loss` | `mse` (yaml `loss_type`) | Training loss: `mse` or `mae` ([details](#choosing-the-metric-and-the-loss)) |
| `--monitor-metric` | `mae` (yaml `monitor_metric`) | Validation metric that picks the best checkpoint and stops training early: `mae`, `mse` or `rmse` |
| `--metrics` | `mae mse` | Metrics reported for validation and test, from `mae mse rmse r2` |
| `--wandb` | off | Log the run to Weights & Biases ([details](#weights--biases-logging)) |
| `--wandb-project`, `--wandb-entity`, `--wandb-name`, `--wandb-group`, `--wandb-tags` | `$WANDB_PROJECT` or `openqdc`, `$WANDB_ENTITY` or your default, derived, none, dataset and model | Weights & Biases settings, see [Choosing the project](#choosing-the-project) |
| `--no-test-results` | off | Do not save the test-set predictions and parity plot (the split of the run is still saved) |
| `--cache-dir` | see [Where files are saved](#where-files-are-saved) | OpenQDC data location |
| `--regen-config` | off | Re-detect the elements and rewrite the dataset yaml |

Models that were run through this pipeline (short runs on subsets):

| Domain | Models |
|---|---|
| graph | `graph/gcn`, `graph/gin`, `graph/gat` |
| simplicial | `simplicial/scn` |
| cell | `cell/cwn` |
| hypergraph | `hypergraph/unignn`, `hypergraph/unignn2`, `hypergraph/edgnn`, `hypergraph/alldeepset`, `hypergraph/allsettransformer` |

Other model configs should work the same way but were not run.

**Hypergraph, simplicial and cell models.** The OpenQDC molecules are graphs, and TopoBench lifts them to the domain of the model automatically, so these models need no extra setup: `--model hypergraph/unignn2` is enough. The default lift for hypergraph models is `HypergraphKHopLifting` with k=1. Another lift can be chosen after `--`, for example `-- transforms=liftings/graph2hypergraph/knn` (the lift configs are in `configs/transforms/liftings/graph2hypergraph/`). That override is applied, but the kNN lift with `hypergraph/unignn2` crashed on Alchemy with `AssertionError: Edge degrees should be positive`; lifts other than the default are untested on these molecules.

**Metrics.** At the end of training TopoBench loads the best checkpoint and tests it on the validation and the test set, printing `val/...` and `test/...` tables with `loss`, `mae` and `mse`. They are in **eV**, on the baseline-subtracted energy. Run output goes to `logs/train/runs/<timestamp>_<pid>/`. A direct `python -m topobench` run trains on GPU 0 (`trainer/default.yaml`); `sub_openqdc.sh` chooses the device for you, and `--` `trainer.devices=[2]` selects another. The per-molecule test predictions and a parity plot are saved too, see [Test-set results](#test-set-results-txt-files-and-parity-plots).

## Training MACE

```bash
python scripts/openqdc/train_mace.py --dataset QM9 --energy-method 0 --max-samples 20000 --epochs 50
```

The script prepares the data, exports `train.xyz`, `valid.xyz` and `test.xyz`, then runs MACE's training (`mace_run_train`, started through `run_mace_train.py` with the Python of this environment, so training and the test evaluation use the same MACE). Extra `mace_run_train` flags go after `--`:

```bash
python scripts/openqdc/train_mace.py --dataset Alchemy -- --num_channels=128 --max_L=1
```

`--num_channels` and `--max_L` replace `--hidden-irreps` and `--max-ell` only as a pair; `--num_channels` alone is silently overridden by `--hidden_irreps`. To change the width, use `--hidden-irreps "128x0e + 128x1o"`.

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | required | OpenQDC dataset name |
| `--energy-method` | from the yaml, else 0 | Energy column, index or name |
| `--target` | from the yaml (`energies`) | `energies` (total energy) or `formation_energies`; written to `REF_energy` in eV |
| `--max-samples`, `--subsample-seed` | from the yaml (all, 0) | Same molecule selection as TopoBench |
| `--valid-fraction`, `--test-fraction`, `--split-seed` | 0.1, 0.1, 0 | Fractions and seed of the split (same defaults as the yaml, `loader.parameters.valid_fraction/test_fraction/split_seed`; TopoBench uses the same split) |
| `--allow-non-total` | off | Export datasets whose energies are not total energies (see QM8) |
| `--epochs`, `--batch-size`, `--valid-batch-size`, `--lr` | 100, 32, 128, 0.01 | Training settings |
| `--patience` | 30 | Early stopping, in epochs |
| `--r-max`, `--max-ell`, `--correlation`, `--num-interactions` | 5.0, 3, 3, 2 | Cutoff and model size |
| `--hidden-irreps` | `64x0e + 64x1o` | Hidden representation |
| `--dtype`, `--device`, `--seed` | float64, cuda if available, 0 | Numeric precision, device, seed |
| `--loss` | `weighted` | Energy loss: `weighted`, `huber` or `l1l2energyforces` ([details](#choosing-the-metric-and-the-loss)) |
| `--huber-delta` | 0.01 | Delta of the Huber loss |
| `--error-table` | `TotalMAE` | Metric shown in the validation log lines and the final tables: `TotalMAE`, `PerAtomMAE`, `TotalRMSE`, `PerAtomRMSE` |
| `--wandb` | off | Log the run to Weights & Biases ([details](#weights--biases-logging)) |
| `--wandb-project`, `--wandb-entity`, `--wandb-name`, `--wandb-group`, `--wandb-tags` | `$WANDB_PROJECT` or `openqdc`, `$WANDB_ENTITY` or your default, the run name, none, dataset and `MACE` | Weights & Biases settings, see [Choosing the project](#choosing-the-project) |
| `--no-test-results` | off | Do not save the test-set predictions and parity plot |
| `--xyz-root`, `--work-dir`, `--cache-dir` | `datasets/openqdc_xyz`, `logs/mace_openqdc`, as in [Where files are saved](#where-files-are-saved) | Where files and runs are saved, and the OpenQDC data |
| `--name` | derived | Run name (see below) |

Settings fixed by the script: `--atomic_numbers` (every element of the export, so that an element that only occurs in the test file cannot crash the evaluation after training), `--model=MACE`, `--compute_forces=False` (energy only, see below), `--E0s=average` (per-element reference energies fitted on the training set), `--scaling=std_scaling` (the default `rms_forces_scaling` needs forces), `--energy_weight=1.0`, `--forces_weight=0.0`, `--amsgrad`, and `--ema` with decay 0.99.

**The `.xyz` files.** Each frame holds the atoms, positions in Angstrom, `pbc=F`, the energy in `REF_energy` (eV) and the OpenQDC molecule index in `openqdc_index`. No forces are written. `export_args.json` records the exact settings.

**Energy-only training.** MACE has no energy-only mode on its command line, so the script combines three things:
- `--compute_forces=False` makes MACE skip the forces, which saves the force gradient (the larger part of a training step). Frames without `REF_forces` also get a force weight of 0, and `--forces_weight=0.0` is set as a second safeguard.
- MACE's energy losses (`weighted`, `l1l2energyforces`) would still read the predicted forces and crash with `TypeError` when none are computed. `run_mace_train.py` therefore skips the force terms of the loss when no forces were computed. The launcher stops with a message if MACE's loss module does not have the expected functions (it is written for MACE 0.3.x).
- The log then says `following quantities will be reported: energy`, and the force columns of the tables show `None`.

Measured on a GPU (Alchemy, 3,000 molecules, 4 epochs): 45.7 seconds per epoch instead of 98.6 with forces on (2.2 times faster), with an identical test MAE (1030.0 meV). To compute forces anyway, add `-- --compute_forces=True` (the extra flag overrides the script's). The `huber` loss is the exception: MACE then switches stress computation on by itself, which needs the same gradient, so there is no speed-up (the log reports `energy, stress`). The script sets `--stress_weight=0.0` for it, so the loss is still only the energy term.

**Run names.** The default run name is `<data settings>_<split settings>_<8-character hash of the model and training settings>`, for example `Alchemy_energies_m0_n2000_s0_split_v0.1_t0.1_s0_4681b105` (`v0.1` is the validation fraction). The same name is used for the run folder and, with `--wandb`, for the W&B run. Runs that differ only in `--lr` or `--hidden-irreps` therefore get their own folder. A second run with **identical** settings refuses to start while the first is active (`Another run is already using ...`); give it its own `--name`. A rerun with the same name after the first has finished moves the old `checkpoints/` folder to `checkpoints_previous_<time>/` (nothing is deleted), because MACE would otherwise pick the highest-epoch checkpoint of the old run as "best"; with `-- --restart_latest` the checkpoints are kept and training continues (the flag does not change the run name). Device and `--name` are not part of the hash.

### Exporting the `.xyz` files on its own

`export_xyz.py` (also called by `train_mace.py` and `prepare_data.py mace`) takes the shared data flags:

```bash
python scripts/openqdc/export_xyz.py --dataset QM9 --energy-method 3 --max-samples 20000
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | required | OpenQDC dataset name |
| `--energy-method`, `--target` | from the yaml, else 0 / `energies` | Energy column and target written to `REF_energy` |
| `--max-samples`, `--subsample-seed` | from the yaml | Molecule subset |
| `--valid-fraction`, `--test-fraction`, `--split-seed` | from the yaml (0.1, 0.1, 0) | The shared split |
| `--allow-non-total` | off | Export datasets whose energies are not total energies (checked on every call, also when the files exist) |
| `--xyz-root`, `--cache-dir` | `datasets/openqdc_xyz`, see [Where files are saved](#where-files-are-saved) | Output and OpenQDC data folders |
| `--overwrite` | off | Export again although a complete export exists (done under the export lock) |

`export_args.json`, written last, holds all settings plus `method_name` and the list of elements of the whole export.

### Test-set metrics in MACE

MACE evaluates the test set once, when training has ended. Everything in this list is done by `mace_run_train`:

1. During training, a checkpoint is saved only when the validation loss improves, and training stops after `--patience` epochs without improvement (default 30) or at `--epochs`.
2. At the end MACE loads the last saved checkpoint, which is the one with the **best validation loss** (the log says `Loaded Stage one model from epoch N for evaluation`).
3. It then evaluates the training, validation and test sets and prints two tables, `Error-table on TRAIN and VALID` and `Error-table on TEST`. The test set is `test.xyz`, which holds `--test-fraction` of the selected molecules (default 10%, never used for training or for choosing the checkpoint).

The tables use `--error_table=TotalMAE` (the default of `train_mace.py`): `MAE E / meV` is the mean absolute error of the **total energy per molecule**, in meV. The force columns show `None` because the data has no forces. Other table types can be chosen with `--error-table`, for example `PerAtomMAE`, `PerAtomRMSE` or `TotalRMSE`.

Where to find the numbers:

| What | Where |
|---|---|
| Final train/validation/test tables | printed to the console and saved in `logs/mace_openqdc/<run name>/logs/<run name>_run-<seed>.log` (search for `Error-table on TEST`) |
| Training and validation metrics per epoch | `logs/mace_openqdc/<run name>/results/<run name>_run-0_train.txt` (one JSON line per entry) |
| Best model | `logs/mace_openqdc/<run name>/<run name>.model` (and `_compiled.model`) |

MACE's own test numbers exist only in that log. After `mace_run_train` has finished, `train_mace.py` evaluates the best model on `test.xyz` once more and saves the per-molecule predictions, the metrics and a parity plot, see [Test-set results](#test-set-results-txt-files-and-parity-plots); it also checks that its MAE agrees with MACE's table.

MACE saves models, checkpoints and logs in `logs/mace_openqdc/<run name>/`. The test evaluation of `train_mace.py` runs inside the run lock. MACE's learning-rate scheduler (`ReduceLROnPlateau`, patience 50) is slower than the early stopping (`--patience` 30), so with the defaults the learning rate normally stays at its initial value; TopoBench uses its own optimiser settings from the config. MACE trains in float64 by default (`--dtype`), TopoBench in float32.

## Choosing the metric and the loss

Both frameworks separate three things: the **loss** that is minimised, the **metric that selects the best checkpoint** (and stops training early), and the **metrics that are reported**.

| | TopoBench | MACE |
|---|---|---|
| Training loss | `--loss mse` or `mae` (default `mse`; yaml `parameters.loss_type`) | `--loss weighted`, `huber` or `l1l2energyforces` (default `weighted`) |
| Best checkpoint and early stopping by | a validation metric: `--monitor-metric mae`, `mse` or `rmse` (default `mae`; yaml `parameters.monitor_metric`); patience 25 epochs | always the validation **loss**, not selectable; patience `--patience` (default 30) |
| Reported metrics | `--metrics mae mse rmse r2` (default `mae mse`) | `--error-table TotalMAE`, `PerAtomMAE`, `TotalRMSE` or `PerAtomRMSE` (default `TotalMAE`) |

**MACE losses** (only the energy term counts, computed on the per-atom energy): `weighted` is the mean squared error, `huber` is the Huber loss with `--huber-delta` (default 0.01), `l1l2energyforces` is the mean absolute error. `--error-table` decides the metric in the per-epoch validation lines of the log (for example `MAE_E` for `TotalMAE`, `MAE_E_per_atom` for `PerAtomMAE`) and in the final tables.

**TopoBench metrics.** The monitored metric is added to the reported metrics automatically. `r2` can be reported but not monitored (TopoBench minimises the monitored metric). The loss and the monitored metric can also be set in the dataset yaml under `parameters`.

```bash
scripts/openqdc/sub_openqdc.sh topobench 3 --dataset Alchemy --model graph/gcn \
    --loss mae --monitor-metric rmse --metrics mae mse rmse r2
scripts/openqdc/sub_openqdc.sh mace 3 --dataset Alchemy --loss huber --huber-delta 0.05 --error-table PerAtomMAE
```

The files with the saved test-set results always contain all metrics, whichever metric you choose here; the choice only changes what the model is trained for and what is shown during training.

### Using the same metric in MACE and TopoBench

Only partly: the two frameworks optimise different quantities by default.

| | TopoBench (default) | MACE |
|---|---|---|
| Loss is computed on | the energy of the **molecule** (error of the total-energy residual) | the energy **per atom** (error divided by the number of atoms) |
| Reported `MAE` | total-energy MAE per molecule, in eV (`test/mae`) | `TotalMAE`: the same quantity, in meV (1 eV = 1000 meV) |
| Reported `RMSE` | `rmse` (eV) | `TotalRMSE` (meV) |
| Per-atom errors | not reported | `PerAtomMAE`, `PerAtomRMSE` |

- **For comparing the results, use the saved `test_metrics.txt`.** Both pipelines write it with the same definitions (computed by the same function from the same kind of data: energy per molecule in eV), so `mae_eV`, `rmse_eV`, `r2`, `mae_per_atom_meV` and the others can be compared directly, whatever loss and metric each model was trained with.
- **To make the training loss comparable**, let TopoBench learn the per-atom energy: `--target per_atom_formation_energies`. Its loss is then per atom like MACE's, so `--loss mse` corresponds to MACE's `--loss weighted` and `--loss mae` to `--loss l1l2energyforces`. With `--monitor-metric mse` (or `mae`) TopoBench then selects the best checkpoint by the same validation quantity as MACE (MACE always uses its validation loss), and `test/mae` is in eV per atom, which is MACE's `PerAtomMAE` divided by 1000. MACE's `huber` loss has no TopoBench counterpart.
- For the same settings both pipelines use the same energies, split and per-element baseline; what still differs is the model and the training procedure (optimiser, schedule, internal scaling).

## Test-set results: .txt files and parity plots

After training, both pipelines save the predictions of the **best model** (best validation loss) on the **test set**, and a parity plot. They are written to a `test_results` folder:

| Pipeline | Folder |
|---|---|
| TopoBench | `logs/train/runs/<timestamp>_<pid>/test_results/` |
| MACE | `logs/mace_openqdc/<run name>/test_results/` |

| File | Content |
|---|---|
| `test_predictions.txt` | one row per test molecule |
| `test_metrics.txt` | error metrics of the whole test set |
| `test_parity.png` | parity plot: predicted against reference energy |
| `split.txt` | the train/valid/test molecules of this run (`openqdc_idx` and split name), written before training starts; see [Splits](#splits-identical-for-mace-and-topobench) |

The `.txt` files are plain text with `#` comment lines at the top (model, dataset, energy column, units), so they load with `numpy.loadtxt`. They are `.txt` (not a data extension) so that Weights & Biases shows them in the browser, in the **Files** tab of the run, without downloading. Add `--no-test-results` to `train_topobench.py` or `train_mace.py` to skip the predictions, metrics and plot (`split.txt` is always written). The run prints `[results] test set: <N> molecules, MAE ...; files in ...` when they have been written.

### `test_predictions.txt`

| Column | Meaning |
|---|---|
| `openqdc_idx` | index of the molecule in the OpenQDC dataset |
| `n_atoms` | number of atoms |
| `reference_eV` | ground-truth energy (the selected target of the selected column, in eV) |
| `prediction_eV` | energy predicted by the model, in eV |
| `error_eV` | `prediction_eV - reference_eV` |

The energies are per molecule, not per atom. With the default target they are total energies (`--target formation_energies` gives formation energies instead; with `per_atom_formation_energies` the per-atom values are multiplied by `n_atoms`, so the files hold formation energies per molecule).

### `test_metrics.txt`

One `name value` pair per line: `n` (molecules), `mae_eV`, `mae_meV`, `rmse_eV`, `r2` (coefficient of determination), `bias_eV` (mean error), `median_abs_error_eV`, `p95_abs_error_eV`, `max_abs_error_eV` and `mae_per_atom_meV`.

### The parity plot

Reference energy on the x axis, predicted energy on the y axis (both in eV, same range), with the line `y = x` for a perfect model. The corner shows the number of molecules, MAE, RMSE and R². Fewer than 3,000 test molecules are drawn as points; from 3,000 on a density plot is used (molecules per bin, log scale), so the points do not cover each other. Because molecular energies span tens to thousands of eV while the errors are a fraction of an eV, a good model sits very close to the line; use `error_eV` in the `.txt` file for the details.

### How the numbers are obtained

- **TopoBench.** TopoBench trains on the baseline-subtracted energy `y`. When training ends, a callback (`OpenQDCTestResults`) loads the best checkpoint and runs it once over the test set. The prediction is converted back to an energy with the stored energy of each molecule: `prediction = energy - (y - y_predicted)`. The MAE equals the `test/mae` that TopoBench prints (checked for `graph/gcn`, `hypergraph/unignn2`, `simplicial/scn` and `cell/cwn`; the molecule index, atom count and energy survive the lifts of all these domains).
- **MACE.** `train_mace.py` loads the best model that `mace_run_train` saved (`<run name>.model`) and predicts every frame of `test.xyz`. It then compares its MAE with the one in MACE's own `Error-table on TEST` and prints `(agrees)` or `(DIFFERS ...)`. MACE decides whether its table is per molecule or per atom (for example the `huber` loss gives a per-atom table), and the check compares with the matching metric; tables of RMSE type are not compared.

Load the results in Python:

```python
import numpy as np

idx, n_atoms, reference, prediction, error = np.loadtxt(
    "test_predictions.txt", unpack=True
)
```

### Notes

- With the same data settings MACE and TopoBench have the same test molecules (see [Splits](#splits-identical-for-mace-and-topobench)), so the `openqdc_idx` columns of their two `test_predictions.txt` files list the same molecules and the errors can be compared line by line after sorting by `openqdc_idx`.
- The files are written only when a run finishes training; a run that is stopped early leaves none.
- Graphs converted by an older version of this code have no `openqdc_idx` and `n_atoms`. Delete the old folder `datasets/graph/openqdc/<Dataset>` so that they are converted again.

## Weights & Biases logging

Add `--wandb` to `train_topobench.py`, `train_mace.py` or `sub_openqdc.sh` to log a run to [Weights & Biases](https://wandb.ai):

```bash
scripts/openqdc/sub_openqdc.sh topobench 3 --dataset Alchemy --model graph/gcn --wandb --wandb-project openqdc
scripts/openqdc/sub_openqdc.sh mace 3 --dataset Alchemy --wandb --wandb-project openqdc --wandb-name alchemy-mace-1
```

**Login.** Log in once with `wandb login` (the key is stored in `~/.netrc`; this machine already has one) or set `WANDB_API_KEY`. `WANDB_MODE=offline` logs to disk without network access; upload such a run later with `wandb sync <run folder>`. `WANDB_MODE=disabled` turns logging off.

| Option | Default | Meaning |
|---|---|---|
| `--wandb-project` | `$WANDB_PROJECT`, else `openqdc` | Project |
| `--wandb-entity` | `$WANDB_ENTITY`, else your default | Team or user that owns the project |
| `--wandb-name` | TopoBench: `<Dataset>_<model with / as ->_m<column>[_n<max_samples>]`, e.g. `QM9_graph-gcn_m0_n20000`; MACE: the run name | Run name |
| `--wandb-group` | none | Group (runs of one comparison or sweep) |
| `--wandb-tags` | TopoBench: dataset and model (`graph-gcn`); MACE: dataset and `MACE` | Tags (names with spaces or special characters are quoted for Hydra automatically) |

**What is logged**

| | TopoBench | MACE |
|---|---|---|
| During training | train and validation metrics per epoch, the metrics of the best epoch, the full config | validation loss and errors per epoch |
| Hyperparameters | all Hydra settings | `hidden_irreps`, `r_max`, `max_ell`, `correlation`, `num_interactions`, `lr`, `batch_size`, `max_num_epochs`, `loss`, `huber_delta`, `seed`, plus `dataset`, `energy_method`, `max_samples`, `target` |
| At the end | the test metrics, and the test-set results | the test-set results |
| Test-set results (`test_results/...`) | the parity plot as an image, all metrics in the run summary, and `test_predictions.txt`, `test_metrics.txt` and `split.txt` uploaded with the run | same |

The local files of a run are in `logs/train/runs/<run>/wandb/` (TopoBench) and `logs/mace_openqdc/<run name>/wandb/` (MACE).

### Choosing the project

The runs of a training go to the project given by `--wandb-project`, under the team or user given by `--wandb-entity`:

```bash
scripts/openqdc/sub_openqdc.sh topobench 3 --dataset Alchemy --model graph/gcn \
    --wandb --wandb-project my-project --wandb-entity my-team
scripts/openqdc/sub_openqdc.sh mace 3 --dataset Alchemy \
    --wandb --wandb-project my-project --wandb-entity my-team
```

- Without `--wandb-entity` the runs go to your default entity (the one of the account you are logged in with). Use the entity when the project belongs to a team.
- A project that does not exist yet is created by the first online run. Open it at `https://wandb.ai/<entity>/<project>`; the training also prints the link at the start of the run.
- To avoid typing it every time, set the defaults once in your shell: `export WANDB_PROJECT=my-project` and `export WANDB_ENTITY=my-team`. A flag always wins over the variable.
- Put the TopoBench and MACE runs you want to compare in the **same project** and give them the same `--wandb-group`, or use the tags (dataset and model are added automatically); W&B can then filter and plot them together.
- The project is used by both training scripts and by `sub_openqdc.sh`, which passes the flags on.

**MACE specifics.** `train_mace.py` always starts MACE through `run_mace_train.py`. For W&B this matters because MACE's own wandb logging crashes on data without forces (it tries to log a placeholder for the missing force error); the launcher turns such values into `None`. `mace_run_train` ends its W&B run when it exits, so the test-set results are added afterwards by resuming the same run (its id is chosen beforehand). In offline mode this leaves two local run folders with the same id; `wandb sync` merges them into one run.

**Tested** only in offline mode, for both pipelines (the recorded runs contain the metrics, config, parity image and test-set summary). No test runs were uploaded to a Weights & Biases account during development.

## Training on entire datasets

Leave `--max-samples` out. Without it, both pipelines use every molecule of the dataset:

```bash
# TopoBench on GPU 3, all of Alchemy (202,579 molecules)
scripts/openqdc/sub_openqdc.sh topobench 3 --dataset Alchemy --model graph/gcn --epochs 100

# MACE on GPU 3, all of Alchemy
scripts/openqdc/sub_openqdc.sh mace 3 --dataset Alchemy --epochs 100
```

Calling the scripts directly works the same way, for example `python scripts/openqdc/train_mace.py --dataset Alchemy --epochs 100`. If you edited `max_samples` in a dataset yaml to a number, set it back to `null`: a yaml value is used when no flag is given.

### How `--max-samples` works

`--max-samples N` trains on a random subset of N molecules; the rest are ignored and the subset is then split into train, validation and test. The choice is reproducible: `--subsample-seed` (default 0) fixes it. If N is larger than the dataset, the whole dataset is used. Both pipelines pick the same molecules for the same N and seed (they use the same random generator), and then split the subset into train/validation/test with the same split (see [Splits](#splits-identical-for-mace-and-topobench)).

- **TopoBench:** `train_topobench.py` passes it as `dataset.loader.parameters.max_samples`; `OpenQDCDataset` draws the subset when it converts the molecules (`topobench/data/datasets/openqdc_dataset.py`). It is part of the converted-graph folder name (`..._n20000_s0`), so different subset sizes never reuse each other's graphs.
- **MACE:** `train_mace.py` does not read it itself. The option is defined in `export_xyz.py` (`add_data_args`, which `train_mace.py` calls on its parser), and `export()` in the same file draws the subset when it writes the `.xyz` files. It is part of the `.xyz` folder name too. For `--max-samples 10000` on QM9 the files are in `datasets/openqdc_xyz/QM9_energies_m0_n10000_s0/split_v0.1_t0.1_s0/`, and MACE then reports about `train=8000, valid=1000` configurations.

### Suggested order for a big dataset

1. **Prepare the data first**, without a GPU, in the background. The training scripts do this automatically, but doing it first keeps a GPU from sitting idle during the download and conversion:
   ```bash
   nohup python scripts/openqdc/prepare_data.py topobench --dataset Alchemy > prep_tb.log 2>&1 &
   nohup python scripts/openqdc/prepare_data.py mace --dataset Alchemy > prep_mace.log 2>&1 &
   ```
2. **Time one epoch on a subset** (for example `--max-samples 20000 --epochs 1`) and extrapolate before committing to a long run.
3. **Start the training** in the background with `nohup ... > run.log 2>&1 &`, on a GPU with enough free memory (`nvidia-smi`). A full dataset does not change the memory per batch, only the number of batches.

### What to expect

Only subsets of up to 30,011 molecules have been run so far, so these are estimates:

- **TopoBench conversion** ran at about 3,300 molecules per second in the tests: roughly 1 minute for Alchemy, 40 seconds for QM9 and several minutes for VQM24 (about 1.1 million molecules). It is cached, so later runs skip it.
- **Training time grows with the number of molecules.** A MACE epoch on 1,600 molecules took about 6.5 seconds on a GPU. Scaling that linearly, a full Alchemy epoch would take roughly 10 to 15 minutes, so 100 epochs would take about a day. TopoBench epochs were not timed on large data.
- **Memory.** The MACE export and the TopoBench conversion both hold all molecules in memory, and both scripts print a warning above 500,000 molecules (this applies to VQM24). MACE also builds neighbor lists for the whole training set at the start, which takes time and memory.
- **CPU runs** are not realistic for full datasets.

### Which datasets are sensible

| Dataset | Molecules | Comment |
|---|---|---|
| Alchemy | 202,579 | Energy labels look right: the safest full-dataset test |
| QM9 | 133,885 | OpenQDC's unit label is wrong (energies about 627 times too large), see [Known limitations](#known-limitations) |
| QM7b | 7,211 | same label problem as QM9 |
| MultixcQM9 | 133,631 | energies are atomization-type, not total (see [Known limitations](#known-limitations)) |
| TMQM | 86,665 | unit labels not checked |
| VQM24 | 1,104,982 | start with a subset (memory, time) |

## The submission script (GPU or CPU)

```bash
scripts/openqdc/sub_openqdc.sh <mace|topobench> <gpu number|cpu> [arguments of the training script]
```

It is modelled on the `sub_*.sh` scripts in the home directory: the device is a command-line argument, thread counts are set for you, and the repo's uv environment (`.venv`) is used.

```bash
scripts/openqdc/sub_openqdc.sh mace 1 --dataset QM9 --max-samples 20000 --epochs 50
scripts/openqdc/sub_openqdc.sh topobench 1 --dataset Alchemy --model graph/gcn --max-samples 20000
scripts/openqdc/sub_openqdc.sh topobench cpu --dataset QM9 --model cell/cwn --max-samples 2000 --epochs 5
ENERGY_METHOD=3 scripts/openqdc/sub_openqdc.sh topobench 1 --dataset QM9 --max-samples 20000
scripts/openqdc/sub_openqdc.sh mace 1 --dataset QM9 --energy-method "B3LYP:def2-svp"
nohup scripts/openqdc/sub_openqdc.sh mace 1 --dataset QM9 > run.log 2>&1 &
```

The second argument decides where the job runs. There is no automatic detection:

| | GPU (a number such as `1`) | CPU (`cpu`) |
|---|---|---|
| Visible devices | `CUDA_VISIBLE_DEVICES` is set to that number; name and memory use of the GPU are printed | all GPUs are hidden |
| MACE | `--device cuda` | `--device cpu` |
| TopoBench | `trainer=gpu` | `trainer=cpu` |
| Threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`) | 1 | 8 |

Everything after the device is passed on to `train_mace.py` or `train_topobench.py`; arguments after a lone `--` go to `mace_run_train` (MACE) or Hydra (TopoBench), for example `-- --restart_latest`. With no arguments, an unknown model name or a bad device the script prints a usage message and stops with a non-zero exit code.

| Environment variable | Meaning |
|---|---|
| `ENERGY_METHOD=<index or name>` | Energy column (a `--energy-method` flag wins over it) |
| `THREADS=<n>` | CPU threads (default 1 on a GPU run, 8 on a CPU run) |
| `MIN_FREE_MB=<n>` | Warn when the chosen GPU has less free memory (default 4000) |
| `OPENQDC_CACHE_DIR` | Where OpenQDC data is stored |

**GPU numbering.** The script sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the number matches `nvidia-smi` (and is checked: a GPU that does not exist stops the script before any data is prepared). For TopoBench the script appends `trainer=<cpu|gpu>` after your arguments, so a `trainer=` override of your own is replaced; for MACE a `--device` of your own wins.

**GPU memory.** The script only warns when the chosen GPU has little free memory; it still starts the job. Several GPUs on this machine have been nearly full, so check `nvidia-smi`.

## Running several jobs at once

Jobs for different models, GPUs and energy columns can run at the same time, also on the same dataset. Everything two jobs could write at the same moment is protected:

| Shared resource | Protection |
|---|---|
| OpenQDC download and loading | one process at a time per dataset (lock file in the cache folder); the others wait and then find the files |
| Interrupted download | files are checked before every load; broken files are removed and downloaded again |
| TopoBench graph conversion | lock: the first job converts, the others wait and reuse the result |
| MACE `.xyz` export | lock; an export counts as complete only when `export_args.json` exists |
| Dataset yaml | lock, and the file is replaced atomically |
| MACE run folder | name includes a hash of the model and training settings; identical settings refuse to share a folder (`run.lock`, held during training and the test evaluation) |
| Cache files (`data.pt`, `baseline.json`, the dataset yaml) | written to a temporary file and renamed |

Locks are released automatically if a job dies. What was tested (always on fresh data): four TopoBench jobs (gcn, gin, gat, scn) on the same dataset and column started together, where exactly one converted the data; two MACE jobs with different learning rates started together, with one export and two run names; two MACE jobs with identical settings, where the second stopped with a clear message; four jobs started together on a dataset that was not downloaded, where exactly one downloaded it; and a download killed midway, which was repaired on the next run.

Not covered: GPU memory (each job needs its own GPU or enough free memory), and OpenQDC sessions you start yourself while a job downloads (the locks only cover these scripts). The split needs no files (it is computed from the OpenQDC indices), so there is nothing to lock.

## Where files are saved

| What | Location |
|---|---|
| Downloaded OpenQDC data | `$OPENQDC_CACHE_DIR`, else `<repo>/datasets/openqdc/<dataset>/preprocessed/` (never `~/.cache` unless you set it) |
| Lock files of the downloads | `<cache>/.<Dataset>.lock`, `<cache>/.<Dataset>.config.lock` |
| TopoBench graphs | `datasets/graph/openqdc/<Dataset>/<tag>/` (tag: see [The per-element baseline](#the-per-element-baseline); with `processed/data.pt`, `baseline.json`, `convert.lock`) |
| TopoBench runs | `logs/train/runs/<timestamp>_<pid>/` (test-set results and `split.txt` in `test_results/`) |
| MACE `.xyz` files | `datasets/openqdc_xyz/<Dataset>_<target>_m<column>_n<N or all>_s<seed>/split_v0.1_t0.1_s0/{train,valid,test}.xyz`, `export_args.json` (settings and the element list) and `export.lock` |
| Split of each run | `<run folder>/test_results/split.txt` (TopoBench and MACE) |
| MACE runs | `logs/mace_openqdc/<run name>/` (models, checkpoints, logs, results, `run.lock`; test-set results in `test_results/`) |

`datasets/` and `logs/` are in `.gitignore`. Move the OpenQDC data to another disk with `export OPENQDC_CACHE_DIR=/absolute/path` **before** starting Python (OpenQDC reads the variable when it is imported).

OpenQDC does not delete cached data by itself. Datasets can be hundreds of GB, so check free space before downloading a large one. The folder names contain the conversion settings but no version: if the conversion code changes, delete the old folders (`datasets/graph/openqdc` and `datasets/openqdc_xyz`) so that the data is made again.

## Generating dataset configs

```bash
python scripts/openqdc/generate_configs.py                              # the 7 default datasets
python scripts/openqdc/generate_configs.py --datasets QM9 --energy-method 3
python scripts/openqdc/generate_configs.py --datasets VQM24 --download  # download, detect elements
python scripts/openqdc/generate_configs.py --all                        # every OpenQDC dataset
```

| Option | Meaning |
|---|---|
| `--datasets` | Datasets to write (default: QM9 MultixcQM9 QM7 QM7b Alchemy VQM24 TMQM) |
| `--all` | Every OpenQDC potential-energy dataset |
| `--energy-method` | Column written into the yaml; default: keep the one already there, else 0 |
| `--download` | Download missing datasets to detect their elements (needs `--datasets`: some datasets are hundreds of GB) |
| `--cache-dir`, `--out-dir`, `--report` | Cache folder, output folder, json report of the collected metadata |

Datasets that are not downloaded get a placeholder yaml (generic element list, marked `NOT DOWNLOADED`) which is replaced as soon as a training run needs the dataset. To use a dataset that has no config yet with TopoBench, just run `train_topobench.py` (or `prepare_data.py topobench`) with `--dataset <name>`: the config is written and the data downloaded automatically. MACE does not need a config.

## Using the data in Python

```python
from topobench.data.utils.openqdc_utils import OpenQDCSource

src = OpenQDCSource("QM9", "datasets/openqdc")  # downloads if needed
print(len(src), src.n_methods, src.method_names[:3])
numbers, charges, positions, energy = src.get(
    0, target="energies", method=0
)  # eV, Angstrom
```

`OpenQDCSource` loads the dataset with `energy_unit="ev"`, `distance_unit="ang"` and checks that the files are complete.

## Viewing molecules, edges and liftings

`topobench/utils/openqdc_viewer.py` draws OpenQDC molecules in 3D (interactive Plotly) with the edges both rules produce, and, if you pass TopoBench models, with the higher-order structures their liftings create. The edges come from `molecular_edges` in `openqdc_utils.py`, the function `OpenQDCDataset` itself calls, so the figure shows exactly the graph TopoBench builds.

```python
from topobench.utils.openqdc_viewer import (
    view_molecules,
    view_molecule,
    view_atoms,
)

# One page with several molecules and the liftings of some TopoBench models
view_molecules(
    "QM9",
    [3, 17, 1234],
    "figures",
    "qm9",  # -> figures/qm9.html
    models=["simplicial/scn", "cell/cwn", "hypergraph/unignn2"],
    bond_scale=[1.1, 1.3, 1.5],
    cutoff=[2.0, 3.0, 5.0],  # lists give menus in the page
)

view_molecule("QM9", index=1234)  # one molecule, opens in a browser
view_atoms(numbers, positions)  # your own atoms (Angstrom)
```

On a machine without a browser write the page and open it elsewhere (`--offline` embeds the Plotly library, about 5 MB, so the page also works without internet):

```bash
python -m topobench.utils.openqdc_viewer --dataset QM9 --indices 3 17 1234 \
    --save-dir figures --base-name qm9 --bond-scale 1.1 1.3 1.5 --cutoff 2 3 5 \
    --models simplicial/scn cell/cwn hypergraph/unignn2
```

### The page

`view_molecules` writes **one** HTML file, `<save_dir>/<base_name>.html`. A bar above the plot has:

| Control | What it does |
|---|---|
| ◀ menu ▶ (or the left/right keys) | switch between the molecules you passed |
| *bond edges*, *cutoff edges* | turn each edge set on or off |
| *energy method* menu | switch the energy shown in the title between all the energy columns of the dataset (for the current geometry) |
| *bond scale*, *cutoff* menus | choose the value of the bond rule and of the cutoff rule (shown when you pass several values) |
| *lifting* menu | switch between the liftings of the models you passed, or *no lifting* |
| *lifted graph* menu | choose whether the lifting is computed on the **bond graph** or on the **cutoff graph** |

Click an entry in the legend to hide or show one trace (for example only the triangles); drag to rotate, scroll to zoom, hover for atom information.

| Feature | How |
|---|---|
| Atom color | one color per element (Jmol colors), legend entry per element with atomic number and count |
| Atom size | proportional to the cube root of the atomic number (`atom_size`): H 7 px, C 13 px, heavy metals up to about 30 px |
| Edge sets | **bond edges** (`bond_scale` × sum of covalent radii; thick, dark) and **cutoff edges** (distance below `cutoff`; thin, light grey), both solid, with their counts in the legend |
| Liftings | the higher-order cells in **light pink**: filled triangles (rank 2 simplices), tetrahedra (rank 3), rings (cell complexes) or hyperedges |
| Parameters | `bond_scale` (default 1.3) and `cutoff` (Å, default 5.0): one number, or a list to get a menu in the page (the same as `--bond-scale 1.1 1.3 1.5` / `--cutoff 2 3 5`); the first value is shown at first. Also `edges` (initial view), `size_scale`, `max_cells`, `energy_method`, `target`, `offline` |
| Command line | `--dataset`, `--indices` (default 0), `--save-dir` (default `.`), `--base-name` (default `molecule`), `--energy-method`, `--bond-scale`, `--cutoff`, `--edges {both,bonds,radius,none}`, `--models`, `--max-cells`, `--cache-dir`, `--offline` |

The title shows the dataset, the molecule index, the number of atoms and the energy of that geometry for the selected energy method, as OpenQDC returns it (QM9 energies carry OpenQDC's unit label, see [The datasets](#the-datasets)). The page stores the energies of **every** energy column for every molecule, so the *energy method* menu switches between them without rereading anything; `energy_method` (index or name, default 0; `--energy-method`) only chooses which one is shown first, and a column that was not computed for a geometry shows *not computed*. The dataset is downloaded first if it is not cached, and every index is checked before the file is written.

### Liftings

Which lifting a model uses is resolved from the Hydra configs exactly as in a training run (`configs/transforms`, `topobench/utils/openqdc_liftings.py`): the run config is composed with the model and the lifting transforms are instantiated. Only transforms of type *lifting* are applied (feature and positional encodings do not change the structure). **The liftings follow the graph.** Every lifting is computed twice per molecule and per value: on the graph of the bond rule (`bond_scale`) and on the graph of the cutoff rule (`cutoff`), exactly the two `edge_method` choices of the dataset configs, and the *lifted graph* menu selects which one is drawn. A dense cutoff graph produces many cells (QM9 molecule with 15 atoms and a 5 Å cutoff: about 400 triangles and 1100 tetrahedra), so at most `max_cells` (default 3000) cells per rank are drawn and the legend then says *first N of M*; a lifting that finds nothing on a graph (for example the clique complex of benzene's bond graph) says so in the legend.

| Model domain | Default lifting | Drawn in pink |
|---|---|---|
| simplicial (`scn`, `san`, `sccn`, ...) | `SimplicialCliqueLifting` | triangles (rank 2) and tetrahedra (rank 3) of the clique complex |
| cell (`cwn`, `can`, ...) | `CellCycleLifting` | one filled polygon per ring (cycle) |
| hypergraph (`unignn`, `unignn2`, `alldeepset`, ...) | `HypergraphKHopLifting` | one hyperedge per atom (the atom and its neighbours) as a filled surface, or a line for two atoms |
| combinatorial (`topotune`) | `GraphTriangleInducedCC` | the cells of rank 2 and higher |
| graph models, `non_relational/mlp`, `pointcloud/deepset` | none | nothing (the model runs on the plain graph) |

Models that share a lifting share one menu entry (for example `simplicial/scn, simplicial/san`). Atoms, edges (rank 1) are not drawn again by the lifting, since the molecular graph shows them. The filled surfaces are drawing aids: a ring is filled as a fan around its centroid, other cells as their convex hull. If a lifting fails on a molecule, a message is printed and that molecule has no pink structure for it. Molecules without triangles (for example benzene) have an empty clique complex of rank 2, which is correct.

## Visualising MACE's cutoff radius and number of layers

`topobench/utils/mace_viewer.py` shows what a MACE model can "see" of a molecule, **from the geometry alone: no model is built, run or trained** (the module does not import MACE or torch). MACE connects every pair of atoms closer than `r_max` and passes messages along these edges once per interaction layer, so after `L` layers the energy of an atom can depend on the atoms within `L` hops of that graph (never farther than `L × r_max`), and each edge counts less towards the cutoff because of a smooth envelope.

```python
from topobench.utils.mace_viewer import view_mace_receptive_field

view_mace_receptive_field("QM9", [3, 17], "figures", "qm9_mace",
                          r_max=[2, 3, 4, 5, 6], layers=[1, 2, 3, 4])   # -> figures/qm9_mace.html
```

```bash
python -m topobench.utils.mace_viewer --dataset QM9 --indices 3 17 --save-dir figures \
    --base-name qm9_mace --r-max 2 3 4 5 6 --layers 1 2 3 4 [--offline]
```

The page is written in a few seconds (it only computes hop counts of the radius graph) and has one 3D view and one plot:

| Element | Meaning |
|---|---|
| ◀ menu ▶ (or the left/right keys) | switch between the molecules you passed (wraps around; the first atom becomes the centre) |
| menus: **centre atom**, **r_max**, **layers** | choose what is shown (a click on an atom also makes it the centre); the page offers exactly the values you pass in `--r-max` and `--layers` (defaults 2 3 4 5 6 and 1 2 3 4), and starts with the first of each; order the lists to start elsewhere |
| atoms | the centre (black, red ring), atoms within `layers` hops of it (coloured by hop: its receptive field), atoms outside it (grey) |
| graph edges | all pairs closer than `r_max`; the shade follows MACE's polynomial cutoff envelope (dark = full weight, light = fading to 0 at `r_max`) |
| spheres | radius `r_max` and radius `layers × r_max` (the distance bound of the receptive field) |
| plot on the right | hop count of every atom against its distance from the centre; dotted lines at `k × r_max` (an atom `k` hops away is always left of the `k`-th line) |
| line above the plot | number of neighbours within `r_max`, atoms in the receptive field, the farthest one and the bound `layers × r_max` |

**Shaded edges.** MACE multiplies the radial features of each edge by a smooth envelope of the distance `d` (a polynomial of exponent 5): 1 at `d = 0` and 0 at `d = r_max`. It makes an edge fade out instead of switching off when an atom crosses the cutoff, which keeps energies and forces continuous. In the page a dark edge has a weight near 1 (short edge) and a light edge is close to `r_max` and hardly used by the model (weight about 0.77 at half of `r_max`, 0.14 at 80 %, 0 at `r_max`). The shade does not change the hop counts: a faded edge still counts as one hop.

The hop counts are exact for MACE's graph; what the page does **not** show is how strongly the model uses each atom, since that depends on the trained weights. The tests check the hop counts on small examples, that an atom `k` hops away is within `k × r_max`, and that the envelope equals the one inside MACE (that last test imports MACE and is skipped without it).

## Comparing MACE and TopoBench

- Use the **same dataset, energy column, target and `--max-samples` / `--subsample-seed`**. Then both see the same molecules and energies.
- TopoBench reports eV (`test/mae`), MACE reports meV (`MAE E / meV` in the final test table; 1 eV = 1000 meV). Both are the mean absolute error of the total energy per molecule after removing a per-element baseline, both measured on the best checkpoint (by validation) on a held-out 10% test set.
- **The splits and the baseline are identical** for the same data settings, see [Splits](#splits-identical-for-mace-and-topobench). This was verified end to end on QM9 (column 3, 2,000 molecules) for `graph/gcn`, `hypergraph/unignn2`, `simplicial/scn`, `cell/cwn` and MACE: identical `split.txt`, identical reference energies in the five `test_predictions.txt` files (equal to OpenQDC's values to 5e-7 eV, the precision of the file), and identical per-element baselines (MACE's `Atomic Energies used` equal TopoBench's `baseline.json` to every printed digit).
- **The inputs differ, as the models do.** MACE sees the element one-hot of each atom and a radius graph (`--r-max`, default 5 Å) built inside the model, and no charge. TopoBench sees the element one-hot plus the formal charge, and the graph of `edge_method` (covalent-radius bonds by default, or a radius graph with `cutoff`); most TopoBench models use only `x` and `edge_index`, not the positions. Use `edge_method: radius` with `cutoff` equal to `--r-max` for the closest graph.

## Troubleshooting

| Problem | What to do |
|---|---|
| `Elements [...] missing from atom_types` | The config was written without data. Run `train_topobench.py --regen-config` |
| Download starts again / cache looks empty | The cached data was removed. Check `ls datasets/openqdc`; the next run downloads again |
| Everything is downloaded twice | Runs used different cache folders. Set `OPENQDC_CACHE_DIR` once and keep it |
| MACE refuses a dataset | Its energies are not total energies (e.g. QM8). Pick another dataset, or pass `--allow-non-total` |
| Errors hundreds of times too large (QM9, QM7b) | OpenQDC's energy-unit label problem (see below) |
| `Checkpoints of an earlier run moved to ...` (MACE) | A run with the same name existed; its checkpoints were set aside so they cannot be picked as the new best model. Delete the `checkpoints_previous_*` folder when you do not need it |
| `Another run is already using ...` (MACE) | A run with identical settings is active. Wait, or give the new run its own `--name` |
| `[wait] ...` | Another job is downloading or preprocessing the same data; the job continues by itself |
| `[repair] ...` | A previous download was interrupted; it is downloaded again, nothing to do |
| CUDA out of memory | Choose a GPU with more free memory (`nvidia-smi`), lower `--batch-size` |
| Training stops immediately for QM7 | Expected: the QM7 data is unusable (all zeros) |
| wandb login prompt | Run `wandb login` once or set `WANDB_API_KEY`; `WANDB_MODE=offline` works without login, and the default `--logger csv` does not use wandb |

## Known limitations

- **Wrong OpenQDC unit labels (not corrected).** The QM9 and QM7b energies are labelled hartree but the numbers are kcal/mol. Checked on methane: the stored value divided by 627.509 gives -40.4789 Ha, the published QM9 U0. Converted with OpenQDC's eV factor they come out about 627 times too large (QM9 molecule 0: -7,223,240 eV instead of about -11,400 eV), and a quick QM9 run gives errors of hundreds of eV. For QM7b the same factor was found by comparing energies with the sum of free-atom energies. Other datasets may have similar errors that were not checked; compare a molecule you know before trusting a new dataset. Alchemy's labels look right.
- **Warnings.** Opening QM9, QM7b or MultixcQM9 logs a one-time warning about the caveats described here (`ENERGY_NOTES` in `openqdc_utils.py`); the values themselves are used unchanged.
- **MultixcQM9** `energies` are atomization-type values (the residual after a per-element fit is under 1 eV, but the numbers are about -4 to -7 eV per atom, not thousands of eV): the default "total energy" target is not a total energy there. Both pipelines train on the same numbers, and the per-element baseline is still a valid linear fit.
- **QM8** columns are excitation energies, not total energies (MACE refuses it unless `--allow-non-total`), and its positions are labelled bohr but look like Angstrom. **ISO17** positions look like bohr although labelled Angstrom. Neither is corrected, and neither has a config in the repo.
- **QM7** cannot be used: the remote OpenQDC copy contains only zeros.
- **Energy-only.** Forces are not used.
- **The split is random per molecule.** Datasets with many conformers of the same molecule (MD trajectories such as RevMD17, MD22, ISO17) leak between train and test; a grouped or scaffold split would need a different `split_molecules`.
- **Energies stored as 0 instead of NaN** for a method that was not computed would pass the finite-energy filter as outliers (only NaN is skipped).
- **Direct `python -m topobench` runs on a fresh dataset** convert and lift under TopoBench's own locks, but the lifted caches of non-graph models (`PreProcessor`) are written without the temporary-file step; a job killed while saving one can leave a truncated cache that has to be deleted by hand. For graph models `PreProcessor` also writes a second, unused copy of the converted data (`<tag>/data.pt`).
- **TopoBench `train/loss`, `val/loss` and `test/loss`** are logged with batch size 1, so they are an unweighted mean over batches; the MAE/MSE metrics used for checkpoint selection are sample-weighted.
- **Large datasets** (millions of molecules) are converted in Python and need a lot of time and memory; use `--max-samples` first.
- A truncated properties file is detected by its last byte only, so a cut-off file could pass by accident with a probability of roughly 1 in 256.
- A download interrupted midway is repaired by downloading the whole dataset again, not only the broken file.

## Development: tests and checks

```bash
.venv/bin/python -m pytest test/data/utils/test_openqdc_utils.py \
    test/data/utils/test_openqdc_integrity.py test/data/utils/test_split_concurrency.py \
    test/utils/test_openqdc_results.py test/data/utils/test_openqdc_split.py test/utils/test_openqdc_viewer.py \
    test/utils/test_openqdc_export.py test/utils/test_openqdc_mace_launcher.py -q
.venv/bin/pre-commit run --files <changed files>     # ruff, ruff-format, numpydoc, yaml, ...
```

The tests cover the helper functions, the integrity check of downloads, the concurrent creation of splits, the saving of test-set results (`.txt` files and plots), the MACE export and the launcher patches, the viewer and the liftings; none of them download anything. `.gitignore` ignores `data/` folders, which includes `test/data/`, so new test files there need `git add -f`.
