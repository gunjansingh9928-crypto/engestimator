# Student Engagement Estimation

End-to-end pipeline that estimates student engagement from webcam video using the
[DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) dataset.
Designed to be lightweight enough for **real-time CPU inference** in classrooms.

*Self project, Mar 2026 – May 2026.*

## Pipeline
1. **Features** (`src/features.py`): MediaPipe Face Mesh + OpenCV give per-frame head pose
   (yaw/pitch/roll), eye aspect ratio (EAR), blink flag and rolling blink rate.
2. **Sequences** (`src/data.py`): 10-second windows at 5 fps (50 x 7) from a stratified
   20% subset of DAiSEE. Train/val/test are split **by subject** to prevent leakage.
3. **Model** (`src/model.py`, `src/train.py`): 1D-CNN in PyTorch with class-weighted
   cross-entropy to handle DAiSEE's skew toward high-engagement labels.
4. **Baseline** (`src/baseline.py`): logistic regression on clip-level summary statistics.
5. **Ablation** (`src/ablation.py`): drops head pose / eye closure / blink features in turn.

## Results (held-out subjects)
| Model | Macro-F1 |
|---|---|
| Logistic regression | 0.24 |
| **1D-CNN** | **0.33** |

Ablation: eye-closure and blink features gave the largest gain. Per-class F1 for all
4 engagement states is written to `results/*.json`.

## Setup
```bash
pip install -r requirements.txt
# edit configs/config.yaml with your DAiSEE paths
python -m src.data          # extract features + build subset
python -m src.baseline      # logistic baseline
python -m src.train         # 1D-CNN
python -m src.ablation      # feature-group ablation
```

## Structure
```
configs/config.yaml   paths and hyperparameters
src/features.py       MediaPipe feature extraction
src/data.py           subset, caching, subject-wise splits
src/model.py          1D-CNN
src/train.py          training + evaluation
src/baseline.py       logistic regression
src/ablation.py       feature ablation
src/evaluate.py       macro / per-class F1
```

## Limitations
DAiSEE labels are noisy and heavily imbalanced, so macro-F1 is modest. Only a 20%
subset was used. Next steps: gaze features, temporal models, full dataset.
