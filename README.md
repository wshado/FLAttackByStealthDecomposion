# Federated Learning Poisoning Attacks

A comprehensive study of targeted model poisoning in federated learning, progressing from single-adversary to multi-adversary coordinated attacks.

## Overview

This project implements and evaluates three milestones of federated learning poisoning:

1. **Baseline (Milestone 1)** — Clean federated learning baseline on Tiny ImageNet
   - ResNet-18 trained with FedAvg across 30 clients
   - Establishes clean accuracy and update norm distributions

2. **Poisoned (Milestone 2)** — Single adversary targeted poisoning
   - One malicious client launches a targeted attack (source class → target class)
   - Demonstrates basic poisoning feasibility against FedAvg aggregation

3. **Distributed (Milestone 3)** — Multi-adversary coordinated attacks
   - Three coordinating adversaries decompose the attack objective
   - Three attack modes:
     - **Naive**: Independent attacks (baseline for comparison)
     - **Coordinated**: Shared shadow delta partitioned across adversaries
     - **Adaptive**: Coordinated + PGD radius tracks honest update norms (stealthier)

## Requirements

### System Requirements
- Python 3.10+
- GPU recommended (NVIDIA or Apple Silicon M-series); CPU fallback supported
- ~8GB RAM (with GPU) or ~16GB RAM (CPU-only)

### Dataset
- **Tiny ImageNet 200** (~200MB)
- Download: [http://cs231n.stanford.edu/tiny-imagenet-200.zip](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- Extract to project root as `./tiny-imagenet-200/`

### Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Choose PyTorch installation for your architecture:

# M2 Mac (ARM64):
pip install torch torchvision --no-cache-dir

# AMD/x86_64 with NVIDIA GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# AMD/x86_64 without GPU (CPU-only):
pip install torch torchvision

# Then install remaining dependencies:
pip install -r requirements.txt

# Set up Jupyter kernel:
python -m ipykernel install --user --name=fl-poisoning --display-name="FL Poisoning"
```

## Quick Start

### Fast Testing (3-5 min per notebook)
All notebooks include a `FAST_MODE` flag for quick validation:

```python
FAST_MODE = True  # Set in cell 8 of each notebook
```

This reduces:
- Clients: 30 → 4
- Rounds: 25 → 3
- Local epochs: 8 → 1
- Batch size: 128 → 64

### Running the Notebooks

Run in order:

```bash
jupyter lab
```

1. **Baseline.ipynb** — Produces:
   - `./models/fedavg_resnet18_baseline.pth`
   - `./histories/baseline.json`

2. **Poisoned.ipynb** — Produces:
   - `./models/fedavg_resnet18_poisoned.pth`
   - `./histories/poisoned.json`

3. **Distributed.ipynb** — Produces:
   - `./models/fedavg_resnet18_distributed.pth`
   - `./histories/distributed.json`
   - Comparison plots in `./figures/`

### Architecture Support

Switch between M2 Mac and AMD (x86_64) by changing one line in cell 0 of each notebook:

```python
DEVICE_TYPE = 'mps'   # M2 Mac (Apple Silicon)
# DEVICE_TYPE = 'cuda'  # AMD with NVIDIA GPU
# DEVICE_TYPE = 'cpu'   # CPU fallback
```

## Key Hyperparameters

All notebooks share core federated learning config (synchronized across Baseline, Poisoned, Distributed):

| Parameter | Value |
|-----------|-------|
| Clients (K) | 30 |
| Clients/round (m) | 6 |
| Rounds (R) | 25 |
| Local epochs (E) | 8 |
| Batch size | 128 |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Dirichlet α | 0.5 |

**Attack-specific:**

| Parameter | Poisoned | Distributed |
|-----------|----------|-------------|
| Adversaries (f) | 1 | 3 |
| Source class → Target | 0 → 50 | 0 → 50 |
| Auxiliary samples | 100 | 100 |
| Boost factor | 10× | 10× |
| PGD radius | — | 10.0 (adaptive: drifts) |

## Objectives & Approach

### Threat Model
- Attacker controls f clients in a federated network of K total clients
- Each round, m clients are randomly selected for training
- Attacker knows the global model state and can craft malicious updates
- Server uses standard FedAvg (no defense)
- Goal: Misclassify source-class images as target class in the final global model

### Attack Strategy

**Poisoned (Single Adversary)**
1. Honest local training on assigned data
2. Adversarial training: retrain on auxiliary data (source class) with target labels
3. Boost the poisoning update by 10×
4. Submit to server; FedAvg aggregates with honest clients

**Distributed (Multi-Adversary, Coordinated Mode)**
1. **Shadow attacker**: Trains on full auxiliary data (target labels only) once per round
2. **Partitioning**: Boosted delta is partitioned by parameter stripe across f adversaries
   - Each adversary only modifies ~1/f of the network parameters
3. **Aggregation**: When all f adversaries are selected, their orthogonal-in-parameter-space updates reconstruct the full delta
4. **Stealth**: Individual adversary updates have low cosine similarity (defeats clustering defenses like FoolsGold)

**Adaptive Mode (Additional)**
- PGD radius adjusts per-round: radius ← 2× median(honest_update_norms_last_round)
- EMA-smoothed with α=0.5
- Adversarial chunks automatically blend into the honest band as training progresses

### Evaluation Metrics

1. **Clean Accuracy** — Test-set accuracy on unmodified images
2. **Attack Success Rate (ASR)** — Fraction of source-class images misclassified as target
3. **Update Norms (L2)** — Per-client and round-mean norms; stealth measured by overlap with honest band
4. **Comparison Plots**
   - Overlay: clean accuracy, ASR, update norms across baseline/poisoned/distributed
   - Stealth scatter: per-client update norm distributions, highlighting adversary visibility

## Project Structure

```
ProjectV2/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── Baseline.ipynb                         # Milestone 1: Clean FL baseline
├── Poisoned.ipynb                         # Milestone 2: Single-adversary attack
├── Distributed.ipynb                      # Milestone 3: Multi-adversary coordinated
├── tiny-imagenet-200/                     # Dataset (download separately)
├── models/                                # Saved model checkpoints
│   ├── fedavg_resnet18_baseline.pth
│   ├── fedavg_resnet18_poisoned.pth
│   └── fedavg_resnet18_distributed.pth
├── histories/                             # Per-round metrics and logs
│   ├── baseline.json
│   ├── poisoned.json
│   └── distributed.json
└── figures/                               # Comparison plots
    ├── comparison_overlay.pdf
    ├── stealth_per_client_norms.pdf
    ├── three_way_overlay.pdf
    └── stealth_three_way.pdf
```

## Performance Notes

**Estimated wall-clock times (M1 Mac, MPS):**
- Baseline: ~1 hour
- Poisoned: ~1 hour
- Distributed (naive mode): ~1.5 hours
- Distributed (coordinated/adaptive): ~2-2.5 hours (includes shadow training overhead)

**FAST_MODE (testing):** 3-5 minutes per notebook (structure intact, smaller scale)

## Reproducing Results

1. Download and extract Tiny ImageNet 200
2. Set `FAST_MODE = True` for quick validation
3. Run notebooks in order (Baseline → Poisoned → Distributed)
4. Switch `ATTACK_MODE` in Distributed to compare `'naive'`, `'coordinated'`, `'adaptive'`
5. Comparison plots auto-generate at notebook end (stored in `./figures/`)

To scale to production:
- Set `FAST_MODE = False`
- Adjust `NUM_CLIENTS`, `NUM_ROUNDS`, `LOCAL_EPOCHS` as needed
- For multi-machine simulation, adapt the training loop to use remote clients

## Key References

- **Baseline federated learning**: [FedAvg (McMahan et al., 2017)](https://arxiv.org/abs/1602.05629)
- **Single-adversary poisoning**: [Bhagoji et al., 2019](https://arxiv.org/abs/1811.03728)
- **Coordinated attacks & stealth**: This project's contribution (Milestone 3)
- **Dataset**: [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.html)

## Notes

- All hyperparameters are deliberately synchronized across the three notebooks so metrics overlay cleanly
- Changing `NUM_CLIENTS`, `NUM_ROUNDS`, `LOCAL_EPOCHS`, etc. requires consistent updates across all notebooks
- The `FAST_MODE` toggle centralizes this — set it once per notebook to maintain alignment
- For defensive experiments, implement defenses in the `Server.aggregate()` method (e.g., clipping, robust aggregation, anomaly detection)
