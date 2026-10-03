"""Drop one feature group at a time and report the change in macro-F1."""
import argparse, os
from .features import FEATURES, GROUPS
from .evaluate import save
from .train import train_cnn
from .utils import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(ap.parse_args().config)
    all_idx = list(range(len(FEATURES)))
    out = {"full": train_cnn(cfg)}
    for name, idx in GROUPS.items():
        keep = [i for i in all_idx if i not in idx]
        out[f"without_{name}"] = train_cnn(cfg, keep=keep)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    save(out, os.path.join(cfg["results_dir"], "ablation_metrics.json"))
