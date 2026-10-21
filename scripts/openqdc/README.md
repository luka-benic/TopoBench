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
8. [Training TopoBench](#training-topobench)
9. [Training MACE](#training-mace)
10. [Choosing the metric and the loss](#choosing-the-metric-and-the-loss)
11. [Test-set results: .txt files and parity plots](#test-set-results-txt-files-and-parity-plots)
12. [Weights & Biases logging](#weights--biases-logging)
13. [Training on entire datasets](#training-on-entire-datasets)
14. [The submission script (GPU or CPU)](#the-submission-script-gpu-or-cpu)
15. [Running several jobs at once](#running-several-jobs-at-once)
16. [Where files are saved](#where-files-are-saved)
17. [Generating dataset configs](#generating-dataset-configs)
18. [Using the data in Python](#using-the-data-in-python)
19. [Comparing MACE and TopoBench](#comparing-mace-and-topobench)
20. [Troubleshooting](#troubleshooting)
21. [Known limitations](#known-limitations)
22. [Development: tests and checks](#development-tests-and-checks)

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
| `scripts/openqdc/export_xyz.py` | The `.xyz` export used by MACE, also usable on its own |
| `scripts/openqdc/generate_configs.py` | Writes the dataset yaml files |
| `configs/dataset/graph/openqdc_<name>.yaml` | One config per dataset (7 in the repo) |
| `topobench/data/utils/openqdc_utils.py` | Shared loader (`OpenQDCSource`), unit constants, integrity check, locks |
| `topobench/data/datasets/openqdc_dataset.py` | `OpenQDCDataset`: OpenQDC molecules as PyG graphs |
| `topobench/data/loaders/graph/openqdc_datasets_loader.py` | `OpenQDCDatasetLoader`, found automatically by TopoBench |
| `topobench/data/utils/split_utils.py` | `random_splitting`, made safe for concurrent jobs |
| `topobench/utils/openqdc_results.py` | Metrics, `.txt` writers and the parity plot (shared by both pipelines) |
| `topobench/callbacks/openqdc_test_results.py` | TopoBench callback that saves the test-set predictions when training ends |
| `scripts/openqdc/mace_test_results.py` | Evaluates the best MACE model on `test.xyz` and saves the same files |
| `scripts/openqdc/run_mace_train.py` | Starts `mace_run_train` with Weights & Biases logging that works without forces |
| `test/data/utils/test_openqdc_*.py`, `test_split_concurrency.py`, `test/utils/test_openqdc_results.py` | Unit tests |

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
| `formation_energies` (default) | total energy minus the sum of OpenQDC's isolated-atom reference energies |
| `energies` | the total energy |
| `per_atom_formation_energies` | formation energy divided by the number of atoms (TopoBench only). TopoBench's losses and metrics are then per atom, like MACE's; the saved test-set results are converted back to energies per molecule |

For datasets without isolated-atom reference energies (QM9, QM8, QM7b) the formation energy equals the total energy.

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

Molecular energies are dominated by the energy of the atoms. With `baseline: per_element` (default) TopoBench fits one energy per element by least squares and trains on the remainder `y = energy - sum(count_z * e0_z)`. The fitted values are written to `baseline.json` next to the graphs. MACE does the same internally (`--E0s=average`), so the errors of both are comparable. `baseline: none` trains on the raw energy. The default is `none` for datasets whose energies are not total energies (QM8).

The baseline is fitted on all converted molecules for TopoBench, and on the training split only for MACE.

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
| MultixcQM9 | 133,631 | 229 | H C N O F | 229 functional/basis-set columns. Tested with TopoBench |
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
- Without a flag, both pipelines use the yaml's value. A flag or variable overrides it **for that run only**; the yaml file is not changed. (The one exception: `--regen-config` together with an explicit column writes that column into the regenerated yaml.)
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
| `target` | `formation_energies` | See [Target](#target) |
| `energy_method` | `0` | The energy column, see above |
| `baseline` | `per_element` (`none` for QM8) | See [The per-element baseline](#the-per-element-baseline) |
| `edge_method` | `bonds` | `bonds` or `radius` |
| `bond_scale` | `1.3` | Bond if `d < bond_scale * (r_i + r_j)` |
| `cutoff` | `5.0` | Angstrom, used when `edge_method: radius` |
| `max_samples` | `null` (all) | Random subset of molecules |
| `subsample_seed` | `0` | Seed of that subset |

Other sections: `parameters` (regression task, `mse` loss, `mae` as monitored metric, `num_features`), `split_params` (random 80/10/10 split, seed 0) and `dataloader_params` (batch size 128). Every parameter can be overridden on the command line with `dataset.loader.parameters.<name>=<value>`.

The `atom_types` list is detected from the data. If a config was written before the data was downloaded it carries a generic element list and a `NOT DOWNLOADED` note; the training script then downloads the dataset and rewrites the file.

## Downloading and preprocessing

Every training run starts with a check and does only what is missing, in this order:

1. The **dataset yaml** exists (otherwise it is written, which downloads the data to detect the elements).
2. The **OpenQDC data** is downloaded and complete.
3. The **preprocessed data** exists for exactly these settings: the TopoBench graphs (for TopoBench) or the `.xyz` files (for MACE). Otherwise they are built.

Progress is printed as `[prepare] ...` lines. The same step can be run alone, for example once before starting many jobs:

```bash
python scripts/openqdc/prepare_data.py topobench --dataset QM9 --energy-method 3 --max-samples 20000
python scripts/openqdc/prepare_data.py topobench --dataset QM9 --loader-param baseline=none
python scripts/openqdc/prepare_data.py mace --dataset QM9 --max-samples 20000
```

`--loader-param KEY=VALUE` (repeatable) sets any other loader parameter of the yaml.

**Interrupted downloads.** OpenQDC writes its files in place and trusts any file that exists, so a download killed midway used to leave a truncated file and a confusing error. Before every load the files are now checked against each other (sizes, number of atoms, molecules and energies, and that the properties file is complete). If a download was cut off, the partial files are removed (`[repair] ...`) and downloaded again.

**Messages you may see**

| Message | Meaning |
|---|---|
| `[prepare] ...` | progress of the check |
| `[wait] ... another job is busy with it, waiting ...` | another job is downloading or preprocessing the same data; this job continues by itself afterwards |
| `[repair] ... incomplete files found` | a previous download was interrupted; the dataset is downloaded again |

Model-specific preprocessing (the simplicial and cell lifts) still happens during training; TopoBench protects it with its own lock.

## Training TopoBench

```bash
python scripts/openqdc/train_topobench.py --dataset QM9 --model graph/gcn --energy-method 0 \
    --max-samples 20000 --epochs 50
```

The script prepares the data (see above), then runs `python -m topobench` with the right overrides. Anything after a lone `--` is passed to Hydra unchanged:

```bash
python scripts/openqdc/train_topobench.py --dataset QM9 --model graph/gcn -- \
    optimizer.parameters.lr=0.001 dataset.loader.parameters.edge_method=radius
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | required | OpenQDC dataset name, case-insensitive |
| `--model` | `graph/gcn` | Any model config under `configs/model/`, e.g. `graph/gin`, `simplicial/scn`, `cell/cwn`, `hypergraph/unignn2` |
| `--energy-method` | from the yaml | Energy column, index or name |
| `--target` | from the yaml | `formation_energies`, `energies` or `per_atom_formation_energies` |
| `--max-samples` | all | Random subset of molecules (recommended for datasets above ~500k molecules) |
| `--subsample-seed` | from the yaml | Seed of that subset |
| `--epochs` | from the config | `trainer.max_epochs` |
| `--batch-size` | 128 | Dataloader batch size |
| `--seed` | from the config | Global seed |
| `--logger` | `csv` | `csv`, `wandb`, `tensorboard`, ... |
| `--loss` | `mse` (yaml `loss_type`) | Training loss: `mse` or `mae` ([details](#choosing-the-metric-and-the-loss)) |
| `--monitor-metric` | `mae` (yaml `monitor_metric`) | Validation metric that picks the best checkpoint and stops training early: `mae`, `mse` or `rmse` |
| `--metrics` | `mae mse` | Metrics reported for validation and test, from `mae mse rmse r2` |
| `--wandb` | off | Log the run to Weights & Biases ([details](#weights--biases-logging)) |
| `--wandb-project`, `--wandb-entity`, `--wandb-name`, `--wandb-group`, `--wandb-tags` | `$WANDB_PROJECT` or `openqdc`, `$WANDB_ENTITY` or your default, derived, none, dataset and model | Weights & Biases settings, see [Choosing the project](#choosing-the-project) |
| `--no-test-results` | off | Do not save the test-set predictions and parity plot |
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

**Metrics.** At the end of training TopoBench loads the best checkpoint and tests it on the validation and the test set, printing `val/...` and `test/...` tables with `loss`, `mae` and `mse`. They are in **eV**, on the baseline-subtracted energy. Run output goes to `logs/train/runs/<timestamp>_<pid>/`. The per-molecule test predictions and a parity plot are saved too, see [Test-set results](#test-set-results-txt-files-and-parity-plots).

## Training MACE

```bash
python scripts/openqdc/train_mace.py --dataset QM9 --energy-method 0 --max-samples 20000 --epochs 50
```

The script prepares the data, exports `train.xyz`, `valid.xyz` and `test.xyz`, then runs the official `mace_run_train`. Extra `mace_run_train` flags go after `--`:

```bash
python scripts/openqdc/train_mace.py --dataset Alchemy -- --num_channels=128 --max_L=1
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | required | OpenQDC dataset name |
| `--energy-method` | from the yaml, else 0 | Energy column, index or name |
| `--target` | `formation_energies` | `formation_energies` or `energies`; written to `REF_energy` in eV |
| `--max-samples`, `--subsample-seed` | all, 0 | Same molecule selection as TopoBench |
| `--valid-fraction`, `--test-fraction`, `--split-seed` | 0.1, 0.1, 0 | Random split done during the export |
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
| `--xyz-root`, `--work-dir` | `datasets/openqdc_xyz`, `logs/mace_openqdc` | Where files and runs are saved |
| `--name` | derived | Run name (see below) |

Settings fixed by the script: `--model=MACE`, `--E0s=average` (per-element reference energies fitted on the training set), `--scaling=std_scaling`, `--amsgrad`, `--ema` with decay 0.99, and energy weight 1.

**The `.xyz` files.** Each frame holds the atoms, positions in Angstrom, `pbc=F`, the energy in `REF_energy` (eV) and the OpenQDC molecule index in `openqdc_index`. No forces are written. `export_args.json` records the exact settings.

**Energy-only quirk.** MACE's loss always contains a force term. The script lets MACE predict forces but gives them weight 0, and the files contain no forces, so the training is correct. Epochs take about three times longer than a pure-energy loop, and the log shows `MAE_F=None`; this is expected.

**Run names.** The default run name is `<data settings>_<split settings>_<8-character hash of the model and training settings>`. Runs that differ only in `--lr` or `--hidden-irreps` therefore get their own folder. A second run with **identical** settings refuses to start while the first is active (`Another run is already using ...`); give it its own `--name`.

### Test-set metrics in MACE

MACE evaluates the test set once, when training has ended. Everything in this list is done by `mace_run_train`:

1. During training, a checkpoint is saved only when the validation loss improves, and training stops after `--patience` epochs without improvement (default 30) or at `--epochs`.
2. At the end MACE loads the last saved checkpoint, which is the one with the **best validation loss** (the log says `Loaded Stage one model from epoch N for evaluation`).
3. It then evaluates the training, validation and test sets and prints two tables, `Error-table on TRAIN and VALID` and `Error-table on TEST`. The test set is `test.xyz`, which holds `--test-fraction` of the selected molecules (default 10%, never used for training or for choosing the checkpoint).

The tables use `--error_table=TotalMAE` (the default of `train_mace.py`): `MAE E / meV` is the mean absolute error of the **total energy per molecule**, in meV. The force columns show `None` because the data has no forces. Other table types can be chosen with `--error-table`, for example `PerAtomMAE`, `PerAtomRMSE` or `TotalRMSE`.

Where to find the numbers:

| What | Where |
|---|---|
| Final train/validation/test tables | printed to the console and saved in `logs/mace_openqdc/<run name>/logs/<run name>_run-0.log` (search for `Error-table on TEST`) |
| Training and validation metrics per epoch | `logs/mace_openqdc/<run name>/results/<run name>_run-0_train.txt` (one JSON line per entry) |
| Best model | `logs/mace_openqdc/<run name>/<run name>.model` (and `_compiled.model`) |

MACE's own test numbers exist only in that log. After `mace_run_train` has finished, `train_mace.py` evaluates the best model on `test.xyz` once more and saves the per-molecule predictions, the metrics and a parity plot, see [Test-set results](#test-set-results-txt-files-and-parity-plots); it also checks that its MAE agrees with MACE's table.

MACE saves models, checkpoints and logs in `logs/mace_openqdc/<run name>/`.

## Choosing the metric and the loss

Both frameworks separate three things: the **loss** that is minimised, the **metric that selects the best checkpoint** (and stops training early), and the **metrics that are reported**.

| | TopoBench | MACE |
|---|---|---|
| Training loss | `--loss mse` or `mae` (default `mse`; yaml `parameters.loss_type`) | `--loss weighted`, `huber` or `l1l2energyforces` (default `weighted`) |
| Best checkpoint and early stopping by | a validation metric: `--monitor-metric mae`, `mse` or `rmse` (default `mae`; yaml `parameters.monitor_metric`); patience 25 epochs | always the validation **loss**, not selectable; patience `--patience` (default 30) |
| Reported metrics | `--metrics mae mse rmse r2` (default `mae mse`) | `--error-table TotalMAE`, `PerAtomMAE`, `TotalRMSE` or `PerAtomRMSE` (default `TotalMAE`) |

**MACE losses** (the force part has weight 0 here, so only the energy term counts, computed on the per-atom energy): `weighted` is the mean squared error, `huber` is the Huber loss with `--huber-delta` (default 0.01), `l1l2energyforces` is the mean absolute error. `--error-table` decides the metric in the per-epoch validation lines of the log (for example `MAE_E` for `TotalMAE`, `MAE_E_per_atom` for `PerAtomMAE`) and in the final tables.

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
- The match is still approximate: the models differ, the splits differ, and the per-element baseline is fitted differently (on all molecules for TopoBench, on the training split for MACE).

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

Both `.txt` files are plain text with `#` comment lines at the top (model, dataset, energy column, units), so they load with `numpy.loadtxt`. They are `.txt` (not a data extension) so that Weights & Biases shows them in the browser, in the **Files** tab of the run, without downloading. Add `--no-test-results` to `train_topobench.py` or `train_mace.py` to skip them. The run prints `[results] test set: <N> molecules, MAE ...; files in ...` when they have been written.

### `test_predictions.txt`

| Column | Meaning |
|---|---|
| `openqdc_idx` | index of the molecule in the OpenQDC dataset |
| `n_atoms` | number of atoms |
| `reference_eV` | ground-truth energy (the selected target of the selected column, in eV) |
| `prediction_eV` | energy predicted by the model, in eV |
| `error_eV` | `prediction_eV - reference_eV` |

The energies are per molecule, not per atom. With the default target they are formation energies (identical to total energies for QM9 and QM7b).

### `test_metrics.txt`

One `name value` pair per line: `n` (molecules), `mae_eV`, `mae_meV`, `rmse_eV`, `r2` (coefficient of determination), `bias_eV` (mean error), `median_abs_error_eV`, `p95_abs_error_eV`, `max_abs_error_eV` and `mae_per_atom_meV`.

### The parity plot

Reference energy on the x axis, predicted energy on the y axis (both in eV, same range), with the line `y = x` for a perfect model. The corner shows the number of molecules, MAE, RMSE and R². Up to 3,000 test molecules are drawn as points; above that a density plot is used (molecules per bin, log scale), so the points do not cover each other. Because molecular energies span tens to thousands of eV while the errors are a fraction of an eV, a good model sits very close to the line; use `error_eV` in the `.txt` file for the details.

### How the numbers are obtained

- **TopoBench.** TopoBench trains on the baseline-subtracted energy `y`. When training ends, a callback (`OpenQDCTestResults`) loads the best checkpoint and runs it once over the test set. The prediction is converted back to an energy with the stored energy of each molecule: `prediction = energy - (y - y_predicted)`. The MAE equals the `test/mae` that TopoBench prints (checked for `graph/gcn`, `hypergraph/unignn2`, `simplicial/scn` and `cell/cwn`; the molecule index, atom count and energy survive the lifts of all these domains).
- **MACE.** `train_mace.py` loads the best model that `mace_run_train` saved (`<run name>.model`) and predicts every frame of `test.xyz`. It then compares its MAE with the one in MACE's own `Error-table on TEST` (when `--error-table` is `TotalMAE`) and prints `(agrees)` or `(DIFFERS ...)`.

Load the results in Python:

```python
import numpy as np

idx, n_atoms, reference, prediction, error = np.loadtxt(
    "test_predictions.txt", unpack=True
)
```

### Notes

- The test sets of MACE and TopoBench are different molecules (their splits differ, see [Comparing MACE and TopoBench](#comparing-mace-and-topobench)). `openqdc_idx` identifies a molecule across runs, so results of two runs can be compared on the molecules they share.
- The files are written only when a run finishes training; a run that is stopped early leaves none.
- Graphs converted by an older version of this code have no `openqdc_idx` and `n_atoms`. The processing version was raised to `v4` for that, so they are converted again (the old `v3` folders are not used any more and can be deleted).

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
| `--wandb-name` | TopoBench: `<Dataset>_<model>_m<column>[_n<max_samples>]`; MACE: the run name | Run name |
| `--wandb-group` | none | Group (runs of one comparison or sweep) |
| `--wandb-tags` | TopoBench: dataset and model; MACE: dataset and `MACE` | Tags |

**What is logged**

| | TopoBench | MACE |
|---|---|---|
| During training | train and validation metrics per epoch, the metrics of the best epoch, the full config | validation loss and errors per epoch |
| Hyperparameters | all Hydra settings | `hidden_irreps`, `r_max`, `max_ell`, `correlation`, `num_interactions`, `lr`, `batch_size`, `max_num_epochs`, `loss`, `huber_delta`, `seed`, plus `dataset`, `energy_method`, `max_samples`, `target` |
| At the end | the test metrics, and the test-set results | the test-set results |
| Test-set results (`test_results/...`) | the parity plot as an image, all metrics in the run summary, and `test_predictions.txt` and `test_metrics.txt` uploaded with the run | same |

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

**MACE specifics.** With `--wandb`, `train_mace.py` starts MACE through `run_mace_train.py`: MACE's own wandb logging crashes on data without forces (it tries to log a placeholder for the missing force error), and the launcher turns such values into `None`. `mace_run_train` ends its W&B run when it exits, so the test-set results are added afterwards by resuming the same run (its id is chosen beforehand). In offline mode this leaves two local run folders with the same id; `wandb sync` merges them into one run.

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

`--max-samples N` trains on a random subset of N molecules; the rest are ignored and the subset is then split into train, validation and test. The choice is reproducible: `--subsample-seed` (default 0) fixes it. If N is larger than the dataset, the whole dataset is used. Both pipelines pick the same molecules for the same N and seed (they use the same random generator), but they split the subset into train/validation/test separately (see [Comparing MACE and TopoBench](#comparing-mace-and-topobench)).

- **TopoBench:** `train_topobench.py` passes it as `dataset.loader.parameters.max_samples`; `OpenQDCDataset` draws the subset when it converts the molecules (`topobench/data/datasets/openqdc_dataset.py`). It is part of the converted-graph folder name (`..._n20000_s0`), so different subset sizes never reuse each other's graphs.
- **MACE:** `train_mace.py` does not read it itself. The option is defined in `export_xyz.py` (`add_data_args`, which `train_mace.py` calls on its parser), and `export()` in the same file draws the subset when it writes the `.xyz` files. It is part of the `.xyz` folder name too. For `--max-samples 10000` on QM9 the files are in `datasets/openqdc_xyz/v4_QM9_formation_energies_m0_n10000_s0/split_v0.1_t0.1_s0/`, and MACE then reports about `train=8000, valid=1000` configurations.

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
| MultixcQM9 | 133,631 | unit labels not checked |
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

**GPU numbering.** The number is the CUDA device number. On this machine `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set, so it matches `nvidia-smi`. If that variable is not set in your shell, the numbers can differ on a machine with different GPU types.

**GPU memory.** The script only warns when the chosen GPU has little free memory; it still starts the job. Several GPUs on this machine have been nearly full, so check `nvidia-smi`.

## Running several jobs at once

Jobs for different models, GPUs and energy columns can run at the same time, also on the same dataset. Everything two jobs could write at the same moment is protected:

| Shared resource | Protection |
|---|---|
| OpenQDC download and loading | one process at a time per dataset (lock file in the cache folder); the others wait and then find the files |
| Interrupted download | files are checked before every load; broken files are removed and downloaded again |
| TopoBench graph conversion | lock: the first job converts, the others wait and reuse the result |
| Train/val/test splits | written to a temporary folder under a lock and renamed when complete |
| MACE `.xyz` export | lock; an export counts as complete only when `export_args.json` exists |
| Dataset yaml | lock, and the file is replaced atomically |
| MACE run folder | name includes a hash of the model and training settings; identical settings refuse to share a folder |

Locks are released automatically if a job dies. What was tested (always on fresh data): four TopoBench jobs (gcn, gin, gat, scn) on the same dataset and column started together, where exactly one converted the data; two MACE jobs with different learning rates started together, with one export and two run names; two MACE jobs with identical settings, where the second stopped with a clear message; four jobs started together on a dataset that was not downloaded, where exactly one downloaded it; and a download killed midway, which was repaired on the next run.

Not covered: GPU memory (each job needs its own GPU or enough free memory), and OpenQDC sessions you start yourself while a job downloads (the locks only cover these scripts). Only `random_splitting` (the split used by these configs) was made safe; the k-fold and stratified split functions are unchanged.

## Where files are saved

| What | Location |
|---|---|
| Downloaded OpenQDC data | `$OPENQDC_CACHE_DIR`, else `<repo>/datasets/openqdc/<dataset>/preprocessed/` (never `~/.cache` unless you set it) |
| Lock files of the downloads | `<cache>/.<Dataset>.lock`, `<cache>/.<Dataset>.config.lock` |
| TopoBench graphs | `datasets/graph/openqdc/<Dataset>/<settings>/` (with `processed/data.pt`, `baseline.json`) |
| TopoBench splits | `datasets/data_splits/<Dataset>_m<column>_n<max_samples>_s<seed>/` |
| TopoBench runs | `logs/train/runs/<timestamp>_<pid>/` (test-set results in `test_results/`) |
| MACE `.xyz` files | `datasets/openqdc_xyz/<settings>/split_v0.1_t0.1_s0/{train,valid,test}.xyz` and `export_args.json` |
| MACE runs | `logs/mace_openqdc/<run name>/` (models, checkpoints, logs, results; test-set results in `test_results/`) |

`datasets/` and `logs/` are in `.gitignore`. Move the OpenQDC data to another disk with `export OPENQDC_CACHE_DIR=/absolute/path` **before** starting Python (OpenQDC reads the variable when it is imported).

OpenQDC does not delete cached data by itself. Datasets can be hundreds of GB, so check free space before downloading a large one. The settings folders contain a version (`v4`), so converted data from an older version of this code is never reused.

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

Datasets that are not downloaded get a placeholder yaml (generic element list, marked `NOT DOWNLOADED`) which is replaced as soon as a training run needs the dataset. To use a dataset that has no config yet, just run a training script with `--dataset <name>`: the config is written and the data downloaded automatically.

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

## Comparing MACE and TopoBench

- Use the **same dataset, energy column, target and `--max-samples` / `--subsample-seed`**. Then both see the same molecules and energies.
- TopoBench reports eV (`test/mae`), MACE reports meV (`MAE E / meV` in the final test table; 1 eV = 1000 meV). Both are the mean absolute error of the total energy per molecule after removing a per-element baseline, both measured on the best checkpoint (by validation) on a held-out 10% test set.
- **The train/validation/test splits still differ**: MACE uses its own random split made during the export, TopoBench uses its split files. Treat a comparison as approximate until they are unified. The saved `test_predictions.txt` files contain `openqdc_idx`, so you can compare the two models on the molecules that are in both test sets.

## Troubleshooting

| Problem | What to do |
|---|---|
| `Elements [...] missing from atom_types` | The config was written without data. Run `train_topobench.py --regen-config` |
| Download starts again / cache looks empty | The cached data was removed. Check `ls datasets/openqdc`; the next run downloads again |
| Everything is downloaded twice | Runs used different cache folders. Set `OPENQDC_CACHE_DIR` once and keep it |
| MACE refuses a dataset | Its energies are not total energies (e.g. QM8). Pick another dataset, or pass `--allow-non-total` |
| Errors hundreds of times too large (QM9, QM7b) | OpenQDC's energy-unit label problem (see below) |
| `Another run is already using ...` (MACE) | A run with identical settings is active. Wait, or give the new run its own `--name` |
| `[wait] ...` | Another job is downloading or preprocessing the same data; the job continues by itself |
| `[repair] ...` | A previous download was interrupted; it is downloaded again, nothing to do |
| CUDA out of memory | Choose a GPU with more free memory (`nvidia-smi`), lower `--batch-size` |
| Training stops immediately for QM7 | Expected: the QM7 data is unusable (all zeros) |
| wandb login prompt | Run `wandb login` once or set `WANDB_API_KEY`; `WANDB_MODE=offline` works without login, and the default `--logger csv` does not use wandb |

## Known limitations

- **Wrong OpenQDC unit labels (not corrected).** The QM9 and QM7b energies are labelled hartree but the numbers are kcal/mol. Checked on methane: the stored value divided by 627.509 gives -40.4789 Ha, the published QM9 U0. Converted with OpenQDC's eV factor they come out about 627 times too large (QM9 molecule 0: -7,223,240 eV instead of about -11,400 eV), and a quick QM9 run gives errors of hundreds of eV. For QM7b the same factor was found by comparing energies with the sum of free-atom energies. Other datasets may have similar errors that were not checked; compare a molecule you know before trusting a new dataset. Alchemy's labels look right.
- **QM8** columns are excitation energies, not total energies (MACE refuses it unless `--allow-non-total`), and its positions are labelled bohr but look like Angstrom. **ISO17** positions look like bohr although labelled Angstrom. Neither is corrected, and neither has a config in the repo.
- **QM7** cannot be used: the remote OpenQDC copy contains only zeros.
- **Energy-only.** Forces are not used.
- **Splits differ** between MACE and TopoBench (see above).
- **Large datasets** (millions of molecules) are converted in Python and need a lot of time and memory; use `--max-samples` first.
- A truncated properties file is detected by its last byte only, so a cut-off file could pass by accident with a probability of roughly 1 in 256.
- A download interrupted midway is repaired by downloading the whole dataset again, not only the broken file.

## Development: tests and checks

```bash
.venv/bin/python -m pytest test/data/utils/test_openqdc_utils.py \
    test/data/utils/test_openqdc_integrity.py test/data/utils/test_split_concurrency.py \
    test/utils/test_openqdc_results.py -q
.venv/bin/pre-commit run --files <changed files>     # ruff, ruff-format, numpydoc, yaml, ...
```

The tests cover the helper functions, the integrity check of downloads, the concurrent creation of splits and the saving of test-set results (`.txt` files and plots); none of them download anything. `.gitignore` ignores `data/` folders, which includes `test/data/`, so new test files there need `git add -f`.
