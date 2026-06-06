# ⚡ PhyTM: Physics-Informed Tri-Modal Time Series Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="python"/>
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg" alt="pytorch"/>
  <img src="https://img.shields.io/badge/transformers-%E2%89%A54.30-yellow.svg" alt="transformers"/>
  <img src="https://img.shields.io/badge/Mamba-optional-9cf.svg" alt="mamba"/>
</p>

> 🧠 **PhyTM** injects the physical spectrum (decay rate **α**, oscillation frequency **ω**) extracted by Dynamic Mode Decomposition (DMD) into three modal paths, unifying **data-driven modeling** with **physical-dynamics constraints**.

| 🧬 Path | Implementation |
|--------|----------------|
| 📝 **Textual** | Physics-informed dual-stream attention Transformer (frozen GPT-2 semantic stream + patch stream; DMD bias injected into the attention logits) |
| 🖼️ **Visual** | GAF image + physics-structured state-space Mamba (state-transition matrix built from the DMD spectrum for long-range stability) |
| 🔢 **Numerical** | Hybrid Transformer-Mamba, preserving raw temporal fidelity |

---

## 📁 Project Structure

```
PhyTM/
├── 🚀 run_benchmark.py        # Single top-level entry point (benchmark + ablation)
├── 📦 requirements.txt
├── 📖 README.md
├── ⚙️  config/                 # Configuration & adaptive policy
│   ├── __init__.py
│   └── default.py             # Config dataclass, dataset/seq_len adaptive policy, ablation cases
├── 🧩 models/                  # Model components
│   ├── __init__.py
│   ├── verbosity.py           # Logging switch (silent by default)
│   ├── ⚡ physics_extractor.py # DMD physical-spectrum extraction (base α/ω + learnable residuals Δα/Δω)
│   ├── 📝 text_modal.py        # Textual path: physics-informed dual-stream attention Transformer
│   ├── 🖼️  image_modal.py       # Visual path: GAF + physics-structured state-space Mamba
│   ├── 🔢 numerical_path.py    # Numerical path: hybrid Transformer-Mamba
│   └── 🧠 phytm.py             # PhyTM main model (tri-modal gated fusion)
├── 📂 data/                    # Data loading
│   ├── __init__.py
│   └── dataset.py             # TimeSeriesDataset
├── 🏋️  engine/                  # Training engine
│   ├── __init__.py
│   └── trainer.py             # run_experiment(): train / val / test for a single run
└── 🧪 experiments/             # Ablation / sensitivity / physics-visualization scripts
```

> 📌 Place data files under the directory pointed to by `--data_root`, named `<dataset>.csv` (with a `date` column + one value column per channel).

---

## ⚙️ Environment Configuration

| Dependency | Version |
|------------|---------|
| 🐍 Python | 3.8.20 |
| 🔥 PyTorch | 2.2.2 |
| 🤗 transformers | ≥4.30 |
| 🔢 numpy | 1.24.3 |
| 📊 pandas | 2.0.3 |
| 🧩 mamba-ssm (optional) | 1.1.3 |
| 🔄 causal-conv1d (optional) | 1.1.3 |

> ⚙️ **The official Mamba CUDA kernel is optional.** If `mamba-ssm` / `causal-conv1d` are not installed, the visual path automatically falls back to a pure-PyTorch parallel scan — no GPU build required. ✅

### 🧩 Environment Validation

Before running the project, please verify that your environment is set up correctly.

#### 1️⃣ Create `check_environment.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment Validation Script for PhyTM
(Physics-Informed Tri-Modal Time Series Forecasting)
---------------------------------------------------------------
Verifies that all required dependencies are correctly installed.
The Mamba CUDA kernel (mamba-ssm / causal-conv1d) is optional: if it
is absent, the visual path falls back to a pure-PyTorch parallel scan.
"""
import sys


def check_environment():
    print("🔍 Checking environment configuration...\n")

    # ── required dependencies ─────────────────────────────────
    try:
        import torch
        import numpy as np
        import pandas as pd
        import transformers
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        sys.exit(1)

    print("✅ Required dependencies are installed correctly!\n")
    print(f"🐍 Python version:       {sys.version.split()[0]}")
    print(f"🔥 PyTorch version:      {torch.__version__}")
    print(f"🤗 transformers version: {transformers.__version__}")
    print(f"🔢 numpy version:        {np.__version__}")
    print(f"📊 pandas version:        {pd.__version__}")

    # ── optional Mamba CUDA kernel ────────────────────────────
    print("\n🧩 Optional Mamba CUDA kernel (falls back to pure-PyTorch if absent):")
    try:
        import mamba_ssm
        import causal_conv1d
        print(f"   ✅ mamba-ssm:     {getattr(mamba_ssm, '__version__', 'unknown')}")
        print(f"   ✅ causal-conv1d: {getattr(causal_conv1d, '__version__', 'unknown')}")
    except ImportError:
        print("   ⚠️  not installed -> visual path uses the pure-PyTorch parallel scan (OK)")

    # ── CUDA information ───────────────────────────────────────
    print("\n💻 CUDA information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        device_id = torch.cuda.current_device()
        print(f"Current device ID: {device_id}")
        print(f"Device name: {torch.cuda.get_device_name(device_id)}")

    print("\n✅ Environment check completed successfully!")


if __name__ == "__main__":
    check_environment()
```

#### 2️⃣ Run the check

```bash
python check_environment.py
```

---

## 🚀 Running

> ✅ **You only need to run the benchmark — no other scripts are required.**
>
> 🔇 By default, only the per-epoch and final **test MSE/MAE** are printed — no GPT, DMD, Adaptive, or Optimizer diagnostics.

```bash
# 🏁 Full benchmark (all datasets × {96, 192, 336, 720})
cd PhyTM
python run_benchmark.py --data_root /path/to/your/data
```

```bash
# 🎯 A subset of datasets and prediction lengths, with fewer epochs
python run_benchmark.py --data_root /path/to/your/data \
    --datasets ETTh1,ETTh2 --pred_lens 96,192 --epochs 5
```


```bash
# 👀 Dry run: list the experiments that would run, without training
python run_benchmark.py --datasets ETTh1,weather --pred_lens 96,192 --dry_run
```

> 💾 Results are written to `benchmark_results.json` and `benchmark_results.md` (flushed after every run, so `Ctrl-C` still preserves completed results).

---

## 🔬 Ablation Cases (`--ablation`)

| Value | Meaning |
|-------|---------|
| `full` | ✅ Complete PhyTM (DMD + Phys-Transformer + Phys-Mamba) |
| `phys_none` | 🚫 Remove physics from both paths (degenerate to standard modules) |
| `phys_no_mamba` | 📝 Physics in the textual path only |
| `phys_no_trans` | 🖼️ Physics in the visual path only |
| `no_text` / `no_image` / `no_num` | ✂️ Remove the textual / visual / numerical path |

---

## 🎯 Training Objective

```
L = L_pred + λ_phys · L_phys
```




---

<p align="center">⚡ <i>Physics in, forecasts out.</i> 🧪</p>
