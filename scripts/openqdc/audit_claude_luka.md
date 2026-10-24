# Audit of the MACE and TopoBench OpenQDC pipelines

Date: 2026-10-24. Scope: `scripts/openqdc/`, the OpenQDC code in `topobench/`, the dataset yamls and the README.

Method: three independent read-only reviews (TopoBench side, MACE side, README against code), plus an end-to-end run of MACE and four TopoBench models on the same data. Findings were checked before they were fixed.

Goal of the audit: for the same data settings, MACE and every TopoBench model must see the same molecules, the same reference energies (eV, positions in Angstrom, OpenQDC's own conversion), the same train/valid/test split and the same per-element baseline.

## 1. End-to-end verification

QM9, energy column 3 (`HF:sto-3g`), 2,000 molecules, 2 epochs, CPU. Models: `graph/gcn`, `hypergraph/unignn2`, `simplicial/scn`, `cell/cwn` and MACE.

| Check | Result |
|---|---|
| `split.txt` of the five runs | identical (train 1600, valid 200, test 200) |
| Test molecules in the five `test_predictions.txt` files | identical |
| Reference energies in those files | identical in all five; equal to OpenQDC's values to 5e-7 eV (the 6-decimal precision of the file) |
| Per-element baseline | MACE's `Atomic Energies used` equals TopoBench's `baseline.json` in every printed digit |
| Result after the fixes below | `graph/gcn` gave the same test MAE as before (569.61 eV) |

## 2. Fixes

### MACE side and launch scripts

| Problem | Fix | Checked |
|---|---|---|
| A rerun with the same run name could evaluate the old run's highest-epoch checkpoint as "best" | the old `checkpoints/` folder is renamed to `checkpoints_previous_<time>` (nothing is deleted) | rerun test |
| `-- --restart_latest` changed the run-name hash, so it never resumed under the default name | the flag is left out of the hash | unit test |
| The `huber` loss silently included a stress term | `--stress_weight=0.0` is passed for it | log shows `stress_weight=0.000` |
| An element that only occurs in the test file crashes MACE after training | the export records all elements (`export_args.json`), the script passes `--atomic_numbers` | fresh export and a short MACE run |
| MACE ignored `max_samples`, `subsample_seed` and `target` from the dataset yaml, so TopoBench and MACE could see different data | they now default to the yaml (a yaml target MACE cannot learn stops with a message) | unit tests |
| `--allow-non-total` was bypassed once an export existed | the refusal is checked on every call | unit test |
| The check that MACE's own MAE agrees with ours was skipped for any seed other than 0 | the log file name uses the seed; the tolerance allows MACE's one-decimal rounding | run with `--seed 1` |
| `--overwrite` removed the export marker outside the lock | removal happens under the lock | read |
| The test evaluation ran outside the run lock | it runs inside | read |
| `sub_openqdc.sh` did not fix the GPU numbering and did not check the GPU number | sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`; stops early if the GPU does not exist | `bash -n` |

### TopoBench side

| Problem | Fix | Checked |
|---|---|---|
| The graph cache folder ignored the element list; a column given as a name or as `"3"` made separate folders | the folder tag has an element hash and the column as an index | unit tests |
| `data.pt` and `baseline.json` were written in place, so a killed job could leave a truncated cache | written to a temporary file and renamed | read |
| Hydra overrides after `--` won in training but not in the data preparation | overrides now win in both; root keys like `trainer=cpu` are no longer treated as loader parameters; `+`/`++` prefixes are handled | unit tests |
| A per-atom target was trained with sum pooling, so the network had to learn the 1/N factor | `model.readout.pooling_type=mean` is set for that target | `graph/gcn` run: per-atom MAE 32.0 eV |
| `--no-test-results` also disabled saving `split.txt` | the split is always saved; the option only skips predictions, metrics and plot | unit test |
| Wandb names, groups and tags with spaces or special characters could break Hydra parsing | values are quoted | unit test |
| `max_samples` below 1 or no finite energy crashed with an unclear error | clear error messages | read |

### Documentation

The README was checked claim by claim. It now has the 21 corrections from the documentation review, a new section for `export_xyz.py`, the real graph folder tag, the viewer command-line flags, lock files, the `trainer=` and `--device` behaviour of `sub_openqdc.sh`, and the limitations below.

## 3. Data problems that remain (not changed)

| Dataset | Issue |
|---|---|
| QM9, QM7b | OpenQDC labels the energies eV, but they are kcal/mol numbers converted as if they were Hartree. The values are about 627.5 times the physical eV, so every error in eV is that much too large. Not corrected, as requested. |
| MultixcQM9 | The `energies` are atomization-type values (about -4 to -7 eV per atom), not total energies. A per-element fit leaves under 1 eV of residual. |

Opening QM9, QM7b or MultixcQM9 now logs a one-time warning (`ENERGY_NOTES` in `openqdc_utils.py`). The values are used unchanged.

## 4. Known limitations (documented in the README)

- The lifted caches of non-graph TopoBench models are written by TopoBench's own `PreProcessor` without a temporary file; a job killed while saving one can leave a truncated cache that has to be deleted by hand. For graph models `PreProcessor` also writes a second, unused copy of the converted data.
- MACE and TopoBench see different inputs: MACE has the element one-hot and its own radius graph (`--r-max`) and no charge; TopoBench has the one-hot plus the formal charge and the bond graph (or a radius graph with `cutoff`), and most of its models use only `x` and `edge_index`, not the positions.
- MACE trains in float64 by default, TopoBench in float32. MACE selects checkpoints by validation loss, TopoBench by the monitored metric. The default losses differ (MACE per atom, TopoBench per molecule); the saved `test_metrics.txt` files use identical definitions.
- MACE's learning-rate scheduler (patience 50) is slower than its early stopping (patience 30), so the learning rate normally stays at its initial value.
- TopoBench logs `train/loss`, `val/loss` and `test/loss` with batch size 1 (an unweighted mean over batches); the metrics used for checkpoint selection are sample-weighted.
- Energies stored as 0 instead of NaN for a method that was not computed pass the finite-energy filter.
- A direct `python -m topobench` run trains on GPU 0 unless `trainer.devices` is overridden.

## 5. Tests

57 OpenQDC tests pass (`test/utils`, `test/data/utils`, selected with `-k openqdc`). New tests cover the split function, the edge rule, the viewer and liftings, the cache folder tag, the checkpoint archive, the element list, loader-parameter parsing, Hydra quoting, the export defaults and the callback option. Pre-commit passes on every changed file.

Nothing from the audit is committed. Test downloads and logs created during the audit were removed.
