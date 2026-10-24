**Both pipelines complete training, checkpoint reload, and evaluation, but the current default results are not trustworthy as a physical-energy benchmark.** Their shared data selection, fixed splits, and total-energy baselines matched in the tested cases. However, both inherit a confirmed QM9/QM7b unit error. VQM24 has a separate, more severe target-consistency problem.

The comparison becomes scientifically interpretable only after validating the dataset’s target and units, preserving the actual split used to fit the baseline, and explicitly accounting for differences in inputs, loss weighting, normalization, and checkpoint selection.

Reviewed branch `ucsb`, commit `5d86eb4cf8f39d64f38fd359ced374e5116d0083`, against base `7682d920d97b36f14d27591754e486085603c0d9`. No project files were changed.

The default comparison is:

| Aspect | TopoBench | MACE |
|---|---|---|
| Data and split | Shared OpenQDC source; subsample, filter nonfinite selected labels, then seeded row split | Same; membership matched |
| Target | Selected energy minus fitted baseline; optional formation or per-atom formation target | Selected total or formation energy |
| Units | Requests eV and Å; inherits incorrect upstream energy declarations | Same |
| Baseline | Training-only least squares; atom counts for extensive targets, atom fractions for per-atom targets | `E0s=average`, training-only least squares using atom counts |
| Inputs | Element one-hot plus formal atomic charge; default inferred bonds. Default GCN does not consume positions or distances | Elements and geometry; 5 Å radius graph; exported charge/spin omitted |
| Default loss | Mean squared **molecular** energy error | Mean squared **per-atom** energy error |
| Checkpoint | Minimum validation MAE, patience 25 | Minimum validation loss, patience 30, EMA enabled; aggregation defect described below |
| Saved test metrics | Shared helper computes restored-target MAE, RMSE, R² and per-atom MAE | Same helper; agreement does not validate units |

Other defaults differ: TB uses batch size 128, 300 epochs, seed 42 and float32; the MACE wrapper uses batch size 32, 100 epochs, seed 0 and float64. MACE also drops incomplete training batches and applies internal energy scaling.

The substantive findings follow, with confirmed behavior distinguished from source-level risks.

1. **Critical: QM9/QM7b energies are converted from the wrong source unit. Confirmed upstream defect affecting both pipelines.**

   [OpenQDC’s QMX declaration](/Users/davidleko/code/TB_work/repos/atomic_topobench/.venv/lib/python3.11/site-packages/openqdc/datasets/potential/qmx.py:51) declares hartree. The [shared loader](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/data/utils/openqdc_utils.py:449) therefore multiplies by approximately `27.211386246`.

   I independently checked the [QM7b source archive](https://zenodo.org/records/3588335): all 76 energy columns declare **kcal/mol**, and all sampled cached values exactly matched the raw values. The correct factor is approximately `0.04336410424 eV/(kcal/mol)`. The resulting scale error is **627.509474**.

   | Record | Raw kcal/mol | Correct eV | Pipeline value labelled eV |
   |---|---:|---:|---:|
   | QM7b row 0, source name `7066` | −433379.127214 | −18793.097649 | −11792846.821581 |
   | QM9 methane, row 4355 | −25400.912063 | −1101.487799 | −691194.029157 |

   For methane, the [original QM9 archive](https://ndownloader.figshare.com/files/3195389) gives `U0 = −40.47893 Ha`, independently confirming the error. Methane is not OpenQDC row 0.

   Positions passed the independent bohr-to-Å check. I found no additional conversion in TB or the XYZ exporter: the problem is the upstream source-unit declaration, not a second conversion downstream.

   **Impact:** energy headers, MAE/RMSE, loss scales, and physical interpretation are wrong. Rescaling reported metrics alone does not undo the effect on optimization.

   **Smallest fix:** correct units at the shared source boundary using verified dataset-specific metadata, invalidate derived caches, and add raw-record regression checks.

2. **Critical: VQM24 combines incompatible methods and duplicate geometries in one target column. Confirmed upstream defect.**

   [OpenQDC’s VQM24 reader](/Users/davidleko/code/TB_work/repos/atomic_topobench/.venv/lib/python3.11/site-packages/openqdc/datasets/potential/vqm24.py:73) concatenates `DFT_all`, `DFT_saddles`, `DFT_uniques`, and `DMC`, taking `Etot` from each and labelling everything `wB97x-D3/cc-pVDZ`.

   These are distinct collections. `DFT_uniques` contains selected conformers from the DFT collection; DMC is a different calculation method. The [authors’ dataset description](https://zenodo.org/records/15442257) explicitly distinguishes them.

   Among 192 inspected records, I found six exact cross-collection geometry duplicates with incompatible targets. For example:

   - Rows `53` and `1094228`: identical two-atom input, energies `−5148.098059722523` and `−26.70953 Ha`.
   - Rows `835958` and `1094190`: identical six-atom input, energies `−5225.4757807228425` and `−39.232666 Ha`.

   The [shared split](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/data/utils/openqdc_utils.py:216) groups only by row number. It cannot keep duplicate structures or conformers together. The full-dataset leakage rate was not measured.

   **Impact:** the model receives identical inputs with radically different labels, and row splits can place related or duplicate records across partitions. This does not establish generalization to unseen molecules.

   **Smallest fix:** reject this combined target until a specific collection/method can be selected; retain molecular identity and use deduplication or grouped splitting.

3. **High: “formation energy” and method labels are not reliable across datasets. Confirmed metadata problems.**

   [OpenQDC’s formation calculation](/Users/davidleko/code/TB_work/repos/atomic_topobench/.venv/lib/python3.11/site-packages/openqdc/datasets/base.py:740) is:

   \[
   F_i=E_i-\sum_a e^{\mathrm{isolated}}_{Z_a,q_a}.
   \]

   It is an isolated-atom reference difference, not a general thermochemical formation enthalpy.

   For QM9 and QM7b, the installed method entries are `NONE`, giving zero isolated-atom references. On all 128 checked records, `formation_energies == energies` exactly.

   [Alchemy’s reader](/Users/davidleko/code/TB_work/repos/atomic_topobench/.venv/lib/python3.11/site-packages/openqdc/datasets/potential/alchemy.py:70) declares ωB97x/6-31G(d), but reads the source’s `U0` column. The [Alchemy paper](https://arxiv.org/html/1906.09427) specifies B3LYP/6-31G(2df,p). Consequently, its formation conversion uses the wrong method’s isolated-atom table.

   [TMQM’s reader](/Users/davidleko/code/TB_work/repos/atomic_topobench/.venv/lib/python3.11/site-packages/openqdc/datasets/potential/tmqm.py:68) advertises def2-TZVP, while the [original dataset documentation](https://github.com/uiocompcat/tmQM) specifies def2-SVP. It selects `Electronic_E`; its XYZ parser also replaces atomic charges with zeros and discards the source’s total-charge/spin information.

   **Impact:** targets and run labels can describe the wrong physical quantity or electronic-structure method.

   **Smallest fix:** validate method/reference metadata per supported dataset, reject unsupported formation conversions, and label the exact selected quantity in results. MultiXC-QM9’s absolute reference convention remains unverified against original calculation logs.

4. **High: TB preparation and training resolve overrides differently. Reproduced branch regression.**

   In [train_topobench.py](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/train_topobench.py:158), parsed Hydra overrides are overwritten by wrapper arguments, including `None`. [prepare_data.py](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/prepare_data.py:103) then removes `None` values and hardcodes its initial data root.

   With only these extra overrides:

   ```text
   dataset.loader.parameters.max_samples=12
   dataset.loader.parameters.target=formation_energies
   dataset.loader.parameters.split_seed=5
   dataset.loader.parameters.energy_method=3
   paths.data_dir=/tmp/requested-data
   ```

   preparation used `max_samples=null`, `target=energies`, `split_seed=0`, and the repository data root. Training received the requested settings. The callback still described energy column **0**, although training used **3**.

   This probe intercepted dataset loading and subprocess execution, so it exercised production argument resolution without launching an unintended full conversion.

   **Impact:** unexpectedly large preparation, redundant caches, and incorrect result/W&B metadata.

   **Smallest fix:** resolve one final Hydra configuration and derive preparation, training, and callback metadata from it.

   MACE has a related source-level risk: [extra arguments are appended last](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/train_mace.py:239), but split recording and post-training evaluation retain the wrapper’s original paths/settings. Overrides such as `--test_file`, `--energy_key`, or `--work_dir` can therefore make training and reported results diverge.

5. **High: changing TB’s split strategy leaks held-out labels into the baseline. Reproduced integration defect.**

   The [dataset fits its baseline](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/data/datasets/openqdc_dataset.py:313) using its own fixed split. Later, the normal [split dispatcher](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/data/utils/split_utils.py:510) allows random, stratified, or k-fold splitting without refitting that baseline.

   On the real 64-record QM7b fixture, switching to the normal random split placed **12 of 16 actual held-out records** in the baseline’s fitting set. Adding `1000` to the actual held-out labels changed:

   - Fitted coefficients by up to `861.447974`.
   - Actual training residual targets by up to `946.674683`.

   With the default fixed split, perturbing held-out labels left coefficients and training targets **bitwise unchanged** for both datasets.

   **Smallest fix:** require the fixed split while using a precomputed fitted baseline, or fit the baseline after the final split is established.

6. **High: MACE’s validation loss aggregation is incorrect in installed version 0.3.16. Reproduced upstream defect.**

   [MACELoss](/Users/davidleko/code/TB_work/repos/atomic_topobench/.venv/lib/python3.11/site-packages/mace/tools/train.py:624) sums batch-mean losses, then divides by the number of configurations.

   For fixed errors `[1, 2, 9]` and atom counts `[1, 2, 3]`, the correct per-atom MSE is `11/3 = 3.666667`. Production evaluation returned:

   | Evaluation batch size | Reported loss |
   |---:|---:|
   | 1 | 3.666667 |
   | 2 | 3.333333 |
   | 3 | 1.222222 |

   MAE and RMSE remained correct. Equal-sized batches introduce a constant scale error; an incomplete final batch changes relative sample weighting and can change checkpoint selection.

   TB has a separate inherited logging issue: [batch_size=1](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/model/model.py:187) makes epoch loss a mean of batch means. Its default `val/mae` checkpoint metric uses the metric accumulator and avoids that particular problem.

   **Smallest fix:** aggregate each mean loss weighted by its actual batch size. Validate or pin a corrected MACE implementation before relying on its checkpoint criterion.

7. **High: MACE’s Huber option is not energy-only. Reproduced integration defect.**

   The wrapper [offers `--loss=huber`](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/train_mace.py:73), but only sets the force weight to zero. MACE activates stress computation, and [its Huber loss](/Users/davidleko/code/TB_work/repos/atomic_topobench/.venv/lib/python3.11/site-packages/mace/modules/loss.py:345) ignores per-configuration missing-stress weights.

   A controlled batch with perfectly predicted energies, no stress labels, and predicted stress equal to one returned loss **0.00995**, rather than zero.

   A real model forward also computed forces and nonzero stress despite `compute_force=False`. Its stress-only contribution was small, approximately `1.10e−11`, but it was still an unintended objective.

   The default weighted loss correctly returned `forces=None`. Its launcher patch is necessary in this installed version: without it, the zero-weight force term still raises a Tensor-minus-None `TypeError`.

   **Smallest fix:** disable Huber for energy-only runs until there is a genuine energy-only Huber path.

8. **High: derived cache identity omits inputs that change the data. Reproduced branch defect.**

   [The graph cache tag](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/data/datasets/openqdc_dataset.py:146) omits ordered `atom_types`, source identity, and conversion/schema version. [XYZ directory identity](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/export_xyz.py:118) similarly omits source/version identity.

   Reversing QM7b’s `atom_types` reused the old cache without processing. The chlorine feature remained in column **5**, while the requested encoding required column **0**. Pointing at a different, nonexistent raw-cache directory also reused existing graphs.

   **Impact:** runs can silently train on stale encodings or pre-correction energies.

   **Smallest fix:** include ordered atom types and a validated source/conversion fingerprint in cache identity. Unit corrections must invalidate both graph and XYZ caches.

9. **High: `test=false` does not prevent test evaluation. Reproduced branch regression.**

   [OpenQDCTestResults.on_fit_end](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/callbacks/openqdc_test_results.py:108) directly loads the best checkpoint and predicts the test set. It bypasses the normal [test flag check](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/run.py:192).

   The lifted smoke run used `test=false` and still wrote predictions for all eight test records. With testing enabled, the callback also duplicates the normal evaluation path.

   Repeated prediction did **not** contaminate metrics: the callback resets its evaluator, and repeated predictions were identical.

   **Smallest fix:** save results from the normal test lifecycle, respecting the test flag, with writes restricted to the appropriate distributed rank.

10. **Medium: completion markers and locks do not fully protect interrupted or concurrent runs. Confirmed marker failure; other risks identified in source.**

   [ensure_export](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/export_xyz.py:155) accepts the existence of `export_args.json` without parsing it or checking the XYZ files. A marker containing only `{` was accepted even though **none of the three XYZ files existed**.

   Graph data and baseline metadata are written directly to final paths. MACE [releases its run lock](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/train_mace.py:255) before post-training evaluation. Config generation [locks by raw-cache directory](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/generate_configs.py:254), although different raw-cache directories can target the same YAML and `.tmp` file.

   **Smallest fix:** validate completion manifests and required artifacts, publish files atomically, and keep locks around the entire operation they protect. Concurrent-process stress tests were not run.

11. **Medium: invalid split fractions and nonfinite geometries are accepted. Reproduced branch defects.**

   [split_molecules](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/data/utils/openqdc_utils.py:216) does not validate fractions or nonempty partitions.

   - Nine records with default fractions produced `9/0/0`.
   - Ten records with validation fraction `−0.1` and test fraction `0.1` produced overlapping partitions with sizes `10/9/0`.
   - A synthetic NaN coordinate was accepted into a production TB graph.

   Nonfinite selected energy labels were handled consistently: replacing two labels with NaN/Inf left 62 records, with identical `48/7/7` splits in both pipelines.

   **Smallest fix:** reject invalid fractions, require the partitions needed by the requested workflow, and reject nonfinite geometry before graph construction/export.

The baseline and loss differences also need precise documentation. Let \(C_{iz}\) be element counts, \(N_i\) atom counts, and \(E_i\) the selected extensive target. TB’s extensive baseline and MACE’s average E0 baseline solve:

\[
b=\arg\min_b\sum_{i\in\mathrm{train}}(E_i-C_i b)^2.
\]

Their coefficients and reconstructed baselines matched exactly on both 64-record fixtures. QM9’s design matrix had rank `4/5`, with fluorine absent; both returned the same zero fluorine coefficient. QM7b had full rank `6/6`.

TB’s per-atom formation option instead solves:

\[
b_{\mathrm{pa}}
=\arg\min_b\sum_{i\in\mathrm{train}}
\left(\frac{F_i-C_i b}{N_i}\right)^2.
\]

That is a different weighted least-squares problem. On QM7b, coefficients differed from the extensive fit by up to `237.334709`, and reconstructed extensive baselines differed by up to `376.665228`.

This is mathematically consistent with an intensive target, but the [README’s matched-loss recipe](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/README.md:425) overstates equivalence. The default optimized losses are:

\[
L_{\mathrm{TB}}=\operatorname{mean}(\Delta E_i^2),
\qquad
L_{\mathrm{MACE}}=\operatorname{mean}\left((\Delta E_i/N_i)^2\right).
\]

MACE’s expression is not pooled normalization by total atoms. Its internal scaling changes the model parameterization, not this loss definition. Rebuilding normalization from the 48 training records gave `57.73820072845918`, matching the saved scale `57.73820072845917`; the shift was zero.

Two reporting details matter:

- Restoring TB’s baseline reconstructs the **selected target**. For formation targets, it does not add isolated-atom references back to obtain total energy.
- Total-energy R² and residual-target R² answer different questions. The GCN smoke run had total R² `0.999999996936`, but residual R² `−0.095031`. Six-significant-digit metric output rounds the former to `1`.

The shared [metrics helper](/Users/davidleko/code/TB_work/repos/atomic_topobench/topobench/utils/openqdc_results.py:31) computes MAE, RMSE, R² and per-atom MAE correctly for supplied arrays. It does not save MSE or per-atom RMSE. Result metadata should identify total/formation/residual targets explicitly.

There are also two smaller operational limitations. [Importing the MACE launcher](/Users/davidleko/code/TB_work/repos/atomic_topobench/scripts/openqdc/run_mace_train.py:105) immediately patches global MACE functions and `wandb.init`; those mutations should occur inside the launcher entry point. Also, “forces are never loaded” is stronger than the implementation: [OpenQDC still accesses forces when present](/Users/davidleko/code/TB_work/repos/atomic_topobench/.venv/lib/python3.11/site-packages/openqdc/datasets/base.py:743), even though these pipelines discard them. `max_samples` is applied after source loading, so it does not bound upstream I/O or conversion memory.

Verification used real records wherever possible. No existing OpenQDC cache was available, so I constructed isolated 64-row caches from actual OpenQDC rows `0–63` using bounded byte-range reads. These were real data, not mocked training records. I additionally downloaded the 7.94 MB QM7b raw archive and a small prefix of the original QM9 archive.

Across 64 QM9 and 64 QM7b records:

- Atomic numbers, atom ordering, explicit hydrogens, row IDs, and retained charge features matched through TB and MACE loading.
- QM7b raw-to-loaded coordinates agreed within `3.13e−7 Å`, consistent with float32 conversion.
- XYZ coordinates differed by at most `5.00e−9 Å`; tolerance was `5.1e−9 Å`.
- Exported and MACE-loaded float64 energies matched exactly; tolerance was `1e−8`.
- All 128 radius graphs matched MACE’s neighbor pairs at 5 Å.
- Both graph constructors produced directed pairs without identity self-edges and used nonperiodic geometry. GCN subsequently adds its own self-loops.
- Subtraction before float32 conversion limited residual rounding error to `2.94e−5` for QM9 and `5.71e−5` for QM7b. Casting total energies first would have introduced errors up to `0.245` and `0.498` in the current pipeline units.

Representative QM7b traces were:

| Row / source ID | Atoms | Raw kcal/mol | Converted = XYZ = MACE | Fitted baseline | Float32 residual |
|---|---:|---:|---:|---:|---:|
| `0 / 7066` | 12 | −433379.127214 | −11792846.821581 | −11792607.248597 | −239.572983 |
| `36 / 0007` | 9 | −73760.038202 | −2007112.889032 | −2007232.515890 | 119.626854 |
| `63 / 0033` | 5 | −106124.224659 | −2887787.267252 | −2887731.549770 | −55.717484 |

These numbers demonstrate consistent transport of the same incorrectly scaled target.

Four genuine one-epoch runs completed: GCN, MACE, lifted UniGNN2, and GCN with per-atom formation targets. Each used **48 training, 8 validation, and 8 test records**, batch size 8, and all seeds zero. Test IDs were:

```text
58, 50, 10, 2, 42, 34, 19, 47
```

Independently recomputed results were:

| Run | Total MAE | Total MSE | Total RMSE | Per-atom MAE | Per-atom MSE |
|---|---:|---:|---:|---:|---:|
| TB GCN | 187.663687 | 60080.275443 | 245.112781 | 19.490142 | 807.174015 |
| MACE | 188.694638 | 59561.449161 | 244.052144 | 19.558291 | 798.926906 |
| TB UniGNN2 | 186.290516 | 60399.247697 | 245.762584 | 19.383305 | 810.627783 |
| TB per-atom GCN | 272.572017 | 116926.985273 | 341.945881 | 25.974477 | 1080.726865 |

**These are current pipeline units, incorrectly labelled eV, not validated physical benchmark scores.** The auxiliary per-atom run also changes the fitted baseline.

TB recorded six finite-gradient steps per run, with maximum parameter changes between `0.00536` and `0.00563`. MACE recorded six optimizer steps with finite optimizer state; a separate production `take_step` directly verified finite gradients and a parameter change of approximately `0.001`.

The saved MACE model exactly matched the saved checkpoint state, including the default EMA path. TB reloaded its selected checkpoint. Lifting preserved IDs, targets, original energies, and atom counts. Prediction text matched reload predictions within `5.1e−7`, consistent with six-decimal serialization.

The execution environment was Python `3.11.14`, OpenQDC `0.1.2`, MACE `0.3.16`, torch `2.3.0`, PyG `2.8.1`, NumPy `1.26.4`, SciPy `1.17.1`, ASE `3.29.0`, e3nn `0.4.4`, Hydra `1.3.2`, torchmetrics `1.9.0`, and pytest `9.1.1`. TB used `lightning 2.4.0`; `pytorch-lightning 2.6.6` was also installed. W&B `0.31.0` was disabled.

The commands below used the temporary 64-row caches described above. Running them against a full cache would select different records.

```bash
export REVIEW_DIR=/private/tmp/openqdc-review.iGWCK8
export REVIEW_REPO=/Users/davidleko/code/TB_work/repos/atomic_topobench
export REVIEW_PY=$REVIEW_REPO/.venv/bin/python

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled WANDB_DISABLED=true
export WANDB_DIR=$REVIEW_DIR/wandb
export WANDB_CACHE_DIR=$REVIEW_DIR/wandb_cache
export WANDB_CONFIG_DIR=$REVIEW_DIR/wandb_config
export WANDB_DATA_DIR=$REVIEW_DIR/wandb_data
export MPLCONFIGDIR=$REVIEW_DIR/matplotlib
export XDG_CACHE_HOME=$REVIEW_DIR/cache
export OPENQDC_CACHE_DIR=$REVIEW_DIR/openqdc
export TMPDIR=$REVIEW_DIR/tmp
export PYTHONPATH=$REVIEW_DIR:$REVIEW_REPO:$REVIEW_REPO/scripts/openqdc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "$REVIEW_REPO"

"$REVIEW_PY" -m pytest -q -p no:cacheprovider \
  --basetemp="$REVIEW_DIR/pytest" \
  test/utils/test_openqdc_export.py \
  test/utils/test_openqdc_mace_launcher.py \
  test/utils/test_openqdc_results.py

"$REVIEW_PY" scripts/openqdc/train_topobench.py \
  --dataset QM7b --model graph/gcn --energy-method 0 --target energies \
  --cache-dir "$OPENQDC_CACHE_DIR" --max-samples 64 --subsample-seed 0 \
  --valid-fraction .125 --test-fraction .125 --split-seed 0 \
  --epochs 1 --batch-size 8 --seed 0 --logger csv -- \
  dataset.loader.parameters.data_dir="$REVIEW_DIR/graphs" \
  paths.data_dir="$REVIEW_DIR/data" paths.log_dir="$REVIEW_DIR/tb_gcn" \
  hydra.run.dir="$REVIEW_DIR/tb_gcn" \
  trainer.accelerator=cpu trainer.devices=1 \
  model.feature_encoder.out_channels=8 extras.print_config=false \
  +callbacks.review_audit._target_=review_hooks.Audit

cd "$REVIEW_DIR"

"$REVIEW_PY" "$REVIEW_REPO/scripts/openqdc/train_mace.py" \
  --dataset QM7b --energy-method 0 --target energies \
  --cache-dir "$OPENQDC_CACHE_DIR" --xyz-root "$REVIEW_DIR/xyz" \
  --max-samples 64 --subsample-seed 0 \
  --valid-fraction .125 --test-fraction .125 --split-seed 0 \
  --epochs 1 --batch-size 8 --valid-batch-size 8 --seed 0 --device cpu \
  --hidden-irreps '8x0e' --max-ell 1 --correlation 2 --num-interactions 1 \
  --work-dir "$REVIEW_DIR/mace" --name smoke -- \
  --num_radial_basis=4 '--radial_MLP=[8]' --num_workers=0
```

The two additional TB commands used the same arguments with these exact changes:

- Lifted run: `--model hypergraph/unignn2`, output directory `tb_lifted`, plus `model.backbone.n_layers=1 test=false`.
- Per-atom run: `--target per_atom_formation_energies --monitor-metric mse`, output directory `tb_peratom`.

The temporary `review_hooks.Audit` callback only recorded gradients, parameter changes, and metadata. Additional production-function probes were run with:

```bash
"$REVIEW_PY" "$REVIEW_DIR/probe_numerics.py"
"$REVIEW_PY" "$REVIEW_DIR/probe_config.py"
"$REVIEW_PY" "$REVIEW_DIR/probe_edgecases.py"
"$REVIEW_PY" "$REVIEW_DIR/verify_smokes.py"
"$REVIEW_PY" "$REVIEW_DIR/probe_vqm24.py"
"$REVIEW_PY" "$REVIEW_DIR/fetch_qm9_anchor.py"
```

**Passed:** 13 existing tests, zero pytest failures; all four training executions completed with checkpoints. The defects above were reproduced separately. Initial sandboxed training launches hit an OpenMP shared-memory restriction; identical commands completed with local process permission.

**Not run:** GPU/DDP, concurrent-process stress tests, convergence experiments, full-dataset duplicate audits, MACE stage-two/save-all-checkpoint variants, or original-log validation of every dataset/method. QM7 training is explicitly blocked by the repository. QM8 is identified as excitation-energy data rather than a total-energy target.

Four test files named in the README do not exist: `test_openqdc_utils.py`, `test_openqdc_integrity.py`, `test_split_concurrency.py`, and `test_openqdc_split.py` under `test/data/utils/`.

Before larger experiments, I would prioritize: verified units and target provenance; consistent configuration resolution and baseline/split handling; correct checkpoint aggregation and test lifecycle; then cache recovery and explicit comparison settings.

All 908 tracked-file hashes matched their starting values. The existing `pyproject.toml` CPU-wheel overrides were preserved, no new ignored project files remained, and the isolated temporary directory was deleted.

Per your AGENTS.md workflow, no fixes were implemented. A focused first patch would be:

Goal: Correct QM9/QM7b units and prevent use of the mixed VQM24 target.

Approach: Apply the verified correction at the shared source boundary, invalidate derived caches, reject mixed VQM24 until collection selection is supported, and add focused regression checks.

Reuse: `OpenQDCSource`, existing cache tags, exporter, and OpenQDC test patterns.

Files: `openqdc_utils.py`, `openqdc_dataset.py`, `export_xyz.py`, `scripts/openqdc/README.md`, and focused OpenQDC tests.

Proceed?
