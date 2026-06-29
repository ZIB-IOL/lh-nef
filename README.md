# [Preprint] Neural Field Tokenizations with Hierarchy and Spatial Locality Priors

[**Alonso Urbano**](https://alonsourbano.com)<sup>1</sup>, [David W. Romero](https://www.davidwromero.xyz)<sup>2</sup>, [Max Zimmer](https://maxzimmer.org)<sup>1</sup>, [Sebastian Pokutta](https://www.pokutta.com)<sup>1,3</sup>

<sup>1</sup> Zuse Institute Berlin (ZIB) &nbsp; <sup>2</sup> Cartesia AI &nbsp; <sup>3</sup> Technische Universität Berlin

Minimal reproduction codebase for the paper.

## Citation

```bibtex
@misc{urbano2026neuralfieldtokenizationshierarchy,
      title={Neural Field Tokenizations with Hierarchy and Spatial Locality Priors}, 
      author={Alonso Urbano and David W. Romero and Max Zimmer and Sebastian Pokutta},
      year={2026},
      eprint={2606.08204},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.08204}, 
}
```

## Structure 

```
src/run.py             single runs
src/launch.py          W&B sweeps / SLURM-style launch (same args + sweep overrides)
src/eval_ckpt.py       evaluate a finished run on its test split
src/cfgs/              all configs
src/tools/             standalone utilities
```

---

## Setup

```bash
pip install -r src/requirements.txt
export LHNEF_DATA_ROOT=/scratch/$USER/data       # datasets
export LHNEF_SAVE_ROOT=/scratch/$USER/nefs_runs  # outputs
```

---

## Stage 1: LH-NeF training (encoder + renderer)

One config per dataset. All Stage-1 entry points live directly in `src/cfgs/`.

| Dataset | Coord / Value | Config |
|---|---|---|
| CIFAR-10 32² | 2D / RGB | `src/cfgs/ae-hip-cifar10-generic-nef.yaml` |
| CelebA-HQ 64² | 2D / RGB | `src/cfgs/ae-hip-celebahq-64-generic-nef.yaml` |
| ImageNet-1k 256² | 2D / RGB | `src/cfgs/ae-hip-imagenet1k-128-generic-nef.yaml` |
| ShapeNet16 voxel occupancy 32³ | 3D / occupancy | `src/cfgs/ae-hip-shapenet16-vox-occ.yaml` |
| ERA5 2m temperature (46×90) | 3D sphere / scalar | `src/cfgs/ae-hip-era5-temperature.yaml` |

Train:

```bash
python src/run.py --cfg src/cfgs/ae-hip-cifar10-generic-nef.yaml
```

Same command shape for every dataset — swap the cfg. CelebA-HQ, ShapeNet16, and ERA5 need data prep first (next subsection).

### Useful things to know

- **HiP variant choice.** Each config picks a HiP shape (`hip_variant` under `encoder.args`). Defined in `src/hip/hip.py` (e.g. `C10_4STAGE_G128_LAST32_C16_K4`). Tokens go *many → few* across blocks; the last pre-bottleneck block is the conditioning representation the renderer attends to. As a general rule when creating new variants: put most compute (self-attention layers) in the last encoder block.
- Training-time subsampling for efficiency is supported via `gt_n_query` on `wrapper_cae_coord_value` (image configs) but full-grid evaluation is mandatory on val/test (otherwise you'll be getting unfair PSNR/l1 loss values).
- **`n_inp` (num of observations input to the tokenizer) must match between train and val/test.** The tokenizer learns a representation tied to a specific input distribution, so if you train feeding it X tokens and then at test time, you feed it say 2*X tokens, you will have a distribution shift. `BaseTrainer._sync_n_inp_across_splits` propagates train `n_inp` to val/test at runtime so that this doesn't happen; the downstream latent extraction trainer also reads `n_inp` from the stage-1 cfg to avoid defaulting to a wrong value.
- **Loss.** L1 by default. ERA5 uses MSE (`loss_cfg.mse_loss: 1.0`) because the downstream benchmark metric is MSE.
- **Selection metric.** PSNR (images), IoU (ShapeNet), MSE (ERA5). Set under `ckpt_select_metric`.

### Data prep

#### CIFAR-10
Auto-downloads to `$LHNEF_DATA_ROOT/cifar10` on first run.

#### CelebA-HQ 64²
Extract the 30k flat folder under `$LHNEF_DATA_ROOT/celeba_hq_256/`. The dataset class deterministically splits 80/10/10 via `val_fraction`/`test_fraction`/`seed`.

#### ImageNet-1k 256²
Standard ImageNet train/val layout under e.g. `/scratch/llm/anon/datasets/pytorch/imagenet`. Hardcoded in the cfg, change `root_path` to your mount.

#### ShapeNet16 voxel occupancy 32³

Download two trees and preprocess:

1. **ShapeNetCore** (voxelizations): from HuggingFace `ShapeNet/ShapeNetCore` (~24 GB).
2. **ShapeNet-Part** (just for the official train/test splits, to pick the subsets from shapenetcore): `shapenetcore_partanno_segmentation_benchmark_v0_normal.tar`.

Both go under `$LHNEF_DATA_ROOT`. Then convert `.binvox` (128³ solid) → 32³ occupancy `.npz`:

```bash
python src/tools/prep_shapenet16_vox_occ.py \
    --shapenetpart_root $LHNEF_DATA_ROOT/shapenetcore_partanno_segmentation_benchmark_v0_normal \
    --shapenet_vox_root $LHNEF_DATA_ROOT/shapenetcore \
    --out_root        $LHNEF_DATA_ROOT/shapenet16_vox_occ_32 \
    --out_res 32 \
    --binvox_name model_normalized.solid.binvox
```

Expect ~12k train / ~1.8k val / ~2.8k test shapes. ~22 skips is normal (ShapeNet-Part references models missing from ShapeNetCore).

#### ERA5 2m temperature

Download the preprocessed 46×90 grid from [Dupont et al. (2021)](https://drive.google.com/drive/folders/1r_sk5auYvllSpDG9ZjroOG0SH0v5kPmM):
- `era5_temp2m_16x_train.zip` (8510)
- `era5_temp2m_16x_val.zip` (1166)
- `era5_temp2m_16x_test.zip` (2420)

Unzip all three side-by-side into `$LHNEF_DATA_ROOT/era5/`. Each `.npz` has `latitude (46,)`, `longitude (90,)`, `temperature (46,90)` in Kelvin. The dataset class normalizes to [0,1] (or [−1,1] if `to_pm1=true`) and maps lat/lon to the 3D unit sphere `(cos θ cos φ, cos θ sin φ, sin θ)`.

### Evaluating a Stage-1 checkpoint

`eval_ckpt.py` reads the run's saved `cfg.yaml`, swaps in the test split, and runs the standard evaluation loop:

```bash
python src/eval_ckpt.py --run_dir /path/to/RUN_DIR --split test
```

For ShapeNet16 specifically there's a dedicated script:

```bash
python src/tools/eval_shapenet16_ckpt.py \
  --ckpt /path/to/RUN_DIR/best-model.pth \
  --data_root $LHNEF_DATA_ROOT/shapenet16_vox_occ_32 \
  --split test --out_dir /tmp/sn16_eval --max_vis 12
```

---

## Stage 2: downstream tasks

All Stage-2 tasks follow a two-step pattern: **(1) extract grouped token representations** from a frozen Stage-1 checkpoint, then **(2) train the downstream model** on those representations.

The extraction step always:
- writes shards + a `manifest.json` under `<stage1_run_dir>/extract/...`
- propagates `n_inp` from the Stage-1 cfg automatically (no manual sync needed)
- uses train-split statistics to normalize val/test (`norm_split: train`)

### Stage 2A: generation (token diffusion + FID)

EDM (Karras et al. 2022) over LH-NeF token latents. Uses the registered `hip_dit` model and `hip_token_dm_trainer`.

#### 1) Extract latents

```bash
python src/run.py --cfg src/cfgs/diffusion/extract/cifar10.yaml \
  --opt extract.stage1_ckpt "/PATH/TO/STAGE1_RUN_DIR/best-model.pth"
```

For CelebA-HQ 64: use `src/cfgs/diffusion/extract/celebahq64.yaml`.

#### 2) Train the diffusion model

```bash
python src/run.py --cfg src/cfgs/diffusion/edm/cifar10.yaml \
  --opt stage1_dir "/PATH/TO/STAGE1_RUN_DIR"
```

A `cifar10_smoke.yaml` sibling runs 10 epochs with full diagnostics — use it to validate the pipeline before launching a long run.

Model shape (`num_groups`, `tokens_per_group`, `token_dim`) is auto-detected from the manifest. The DM run lands inside `stage1_dir/` (controlled by `save_root: from_stage1_dir`).

#### 3) Build the "real images" directory (once per dataset, for FID)

Safest path: export the exact split used by the Stage-1 run, read from its checkpoint:

```bash
python src/tools/make_real_image_dir.py lhnef_ckpt \
  --ckpt "/PATH/TO/STAGE1_RUN_DIR/best-model.pth" \
  --split train --size 32 \
  --out_dir "$LHNEF_DATA_ROOT/cifar10_fid_train_32x32"
```

Use `--size 64` for CelebA-HQ 64.

#### 4) Sample + FID

```bash
python src/run.py --cfg src/cfgs/diffusion/sample/edm_cifar10.yaml \
  --opt stage1_dir "/PATH/TO/STAGE1_RUN_DIR" \
  --opt latents_subdir "extract/latents_cifar10_tc" \
  --opt sample.dm_ckpt "/PATH/TO/DM_RUN_DIR/best-model.pth" \
  --opt sample.n_samples 50000 \
  --opt sample.eval.real_dir "$LHNEF_DATA_ROOT/cifar10_fid_train_32x32"
```

For CelebA-HQ 64: use `edm_celebahq64.yaml` and generate 24k samples (matches training-set size).

### Stage 2B: classification on frozen latents

The classifier we use in the paper is `convnext_grouped_classifier` for 2D image latents and `hip_structured_classifier` for 3D ShapeNet latents.

#### CIFAR-10

```bash
# 1) Extract labeled latents (with 50x augmentations for the augmented functaset).
python src/run.py --cfg src/cfgs/classification/extract_cifar10.yaml \
  --opt extract.stage1_ckpt "/PATH/TO/STAGE1_RUN_DIR/best-model.pth"

# 2) Train classifier.
python src/run.py --cfg src/cfgs/classification/classify_cifar10.yaml \
  --opt stage1_dir "/PATH/TO/STAGE1_RUN_DIR"
```

The classifier selects the best checkpoint by `val/acc` and reports `test/acc` at the end.

#### ShapeNet16

```bash
# 1) Extract labeled latents (16-class synset IDs mapped to 0–15).
python src/run.py --cfg src/cfgs/classification/extract_shapenet16.yaml \
  --opt extract.stage1_ckpt "/PATH/TO/STAGE1_RUN_DIR/best-model.pth"

# 2) Train classifier (hip_structured_classifier).
python src/run.py --cfg src/cfgs/classification/classify_shapenet16.yaml \
  --opt stage1_dir "/PATH/TO/STAGE1_RUN_DIR"
```

### Stage 2C: ERA5 temporal forecasting

```bash
# 1) Extract per-timestep latents.
python src/run.py --cfg src/cfgs/era5_forecasting/extract_era5.yaml \
  --opt extract.stage1_ckpt "/PATH/TO/STAGE1_RUN_DIR/best-model.pth"

# 2) Train the forecaster on consecutive-hour latent pairs (no cross-split leakage).
python src/run.py --cfg src/cfgs/era5_forecasting/forecast_era5_hip.yaml \
  --opt stage1_dir "/PATH/TO/STAGE1_RUN_DIR"
```

When the Stage-1 checkpoint is in the cfg, the forecaster loads the frozen renderer and trains in **function space** (decoded temperature MSE, not latent MSE). Eval reports `Tt_mse` (recon) and `Tt1_mse` (forecast); checkpoint selection minimizes `Tt1_mse`.

---

## Sweeps (W&B)

Sweep configs for wandb live under `src/cfgs/sweeps/<dataset>/`.

---

## Tests

Three tiers of regression coverage. Run `pytest` from the repo root.

| Tier | What it checks | Cost | Where |
|---|---|---|---|
| 1 — smoke (`tests/test_smoke.py`) | configs resolve, models build, forward pass returns finite loss | CPU, ~10s | local + GitHub Actions on every PR |
| 2 — overfit (`tests/test_overfit.py`) | Stage-1 model can overfit 4 synthetic images to PSNR > 25 dB in 300 iters (catches broken encoder/renderer/loss/optimizer wiring) | 1 GPU, ~1 min | local |
| 4 — regression (`tests/test_regression.py`) | runs `eval_ckpt.py` on frozen paper checkpoints, asserts metric within tolerance of recorded baseline | 1 GPU, ~1 min/ckpt | local |

```bash
pytest                            # all tiers (GPU tests auto-skip if no CUDA)
pytest tests/test_smoke.py        # Tier 1 only
pytest -m "not gpu"               # CPU-only subset
pytest -m regression              # Tier 4 only
```

#### Configuring Tier 4 baselines

Edit `tests/regression_baselines.yaml` once per paper checkpoint:

```yaml
cifar10:
  ckpt: /scratch/.../RUN_DIR     # contains best-model.pth + cfg.yaml
  split: test
  metric: psnr
  expected: 27.65                 # paper test/psnr
  tol: 0.05                       # ±dB
```

Entries with empty `ckpt` (or non-existent paths) skip — safe to commit before recording baselines. To record a baseline:

```bash
python src/eval_ckpt.py --run_dir /PATH/TO/RUN_DIR --split test --no_vis
# copy reported psnr/mse/iou into tests/regression_baselines.yaml
```

#### CI

GitHub Actions (`.github/workflows/ci.yml`) runs Tier 1 on every PR and push to `main`, on an Ubuntu CPU runner with CPU-only PyTorch wheels. No GPU runners are configured, so Tiers 2 and 4 are intentionally local-only — run them before pushing a branch that touches Stage-1 modeling code.

