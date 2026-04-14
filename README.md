# Chinese Traditional Music Instrument Classification with Pretrained Audio Models

## Repository Structure

```
dataset_cleaning/           Data cleaning pipeline & frozen outputs
  pipeline.py               Main cleaning pipeline
  freeze_config.yaml        Frozen class/threshold/seed decisions
  output/                   Frozen manifests (input to experiments)

instrument_cls_experiments/
  configs/                  All experiment configurations
  src/
    baselines/              MFCC+SVM baseline
    models/                 CLAP & MERT model wrappers
    data/                   Dataset & dataloader utilities
    train/                  Training harnesses
    eval/                   Metrics, confusion, aggregation
    runners/                Experiment entry points
    analysis/               Figure & table generation scripts
  reports/
    tables/                 Aggregated result CSVs
    figures/                Generated paper figures (PDF)

paper_figs_tables/          LaTeX table generation helpers
requirements.txt            Python dependencies
```

## Setup

**Python**: >= 3.9, < 3.13

```bash
pip install -r requirements.txt
```

**Pretrained models**: download to `model_cache/huggingface/` (offline loading):

| Model | HuggingFace ID |
|-------|----------------|
| MERT  | `m-a-p/MERT-v1-95M` |
| CLAP  | `laion/clap-htsat-unfused` |

**System dependency** (data cleaning only): `fpcalc` (Chromaprint)

## Data Preparation

1. Obtain raw audio from [CCMusic](https://github.com/ccmusic-database) and [ChMusic](https://github.com/haoranweiutd/chmusic).

2. Copy `dataset_cleaning/config.yaml.example` to `dataset_cleaning/config.yaml` and fill in local paths.

3. Run the cleaning pipeline:

```bash
python dataset_cleaning/pipeline.py dataset_cleaning/config.yaml
```

This produces frozen manifests and 5-second segments under `dataset_cleaning/output/`.

**Alternatively**, skip step 3 and use the pre-generated manifests already included in `dataset_cleaning/output/` to reproduce experiments directly (raw audio files are still required for actual training).

## Running Experiments

Experiments are organized into 4 phases. Run them **sequentially** — each phase depends on the previous one.

### Phase 0: Smoke Test

```bash
python instrument_cls_experiments/src/runners/run_data_contract_audit.py
python instrument_cls_experiments/src/runners/run_audio_io_audit.py
python instrument_cls_experiments/src/runners/run_smoke_seed1.py
```

### Phase 1: Transfer Comparison (MFCC+SVM, CLAP-ZS, CLAP-LP, MERT-LP)

```bash
python instrument_cls_experiments/src/runners/run_phase1_gate_check.py
python instrument_cls_experiments/src/runners/run_exp01_transfer_all.py
```

### Phase 2: Adaptation Strategies (Linear Probe, LoRA r=4/8, Full FT)

```bash
python instrument_cls_experiments/src/runners/run_phase2_gate_check.py
python instrument_cls_experiments/src/runners/run_exp02_adaptation_all.py
```

### Phase 3: Data Efficiency (10%/25%/50%/100% training data)

```bash
python instrument_cls_experiments/src/runners/run_phase3_gate_check.py
python instrument_cls_experiments/src/runners/run_exp03_data_efficiency_all.py
```

### Phase 4: Layer-wise Probe & Qualitative Analysis

```bash
python instrument_cls_experiments/src/runners/run_phase4_gate_check.py
python instrument_cls_experiments/src/runners/run_phase4_all.py
```

Each `run_*_all.py` script runs all methods across 3 random seeds (3407, 2027, 9413) and writes aggregated results to `reports/tables/` and `reports/figures/`.

## Verifying Results

Pre-computed aggregated tables are included under `instrument_cls_experiments/reports/tables/`. Key files and their corresponding paper content:

| File | Paper |
|------|-------|
| `exp01_transfer_summary_mean_std.csv` | Table V |
| `exp02_adaptation_summary_mean_std.csv` | Table VII |
| `exp02_adaptation_cost_summary_mean_std.csv` | Table VIII |
| `exp03_data_efficiency_summary_mean_std.csv` | Table IX |
| `exp04_layer_probe_summary_mean_std.csv` | Table X |
| `exp04_per_class_all_methods.csv` | Table VI |

Generated figures are under `instrument_cls_experiments/reports/figures/`.
