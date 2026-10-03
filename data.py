"""Build the stratified 20% DAiSEE subset and cache 10 s feature sequences."""
import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from tqdm import tqdm
from .features import extract_clip_features
from .utils import load_config


def build_manifest(cfg):
    labels = pd.read_csv(cfg["labels_csv"])
    labels.columns = [c.strip() for c in labels.columns]
    labels["ClipID"] = labels["ClipID"].astype(str)
    labels["ClipID"] = labels["ClipID"].where(labels["ClipID"].str.endswith(".avi"),
                                              labels["ClipID"] + ".avi")
    labels["subject"] = labels["ClipID"].str[:6]
    paths = {os.path.basename(p): p for p in
             glob.glob(os.path.join(cfg["daisee_root"], "**", "*.avi"), recursive=True)}
    labels["path"] = labels["ClipID"].map(paths)
    labels = labels.dropna(subset=["path"])
    subset, _ = train_test_split(labels, train_size=cfg["subset_frac"],
                                 stratify=labels["Engagement"], random_state=cfg["seed"])
    return subset.reset_index(drop=True)


def build_sequences(cfg):
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    m = build_manifest(cfg)
    X, y, g = [], [], []
    for _, r in tqdm(m.iterrows(), total=len(m)):
        f = os.path.join(cfg["cache_dir"], r["ClipID"] + ".npy")
        if os.path.exists(f):
            feats = np.load(f)
        else:
            feats = extract_clip_features(r["path"], cfg["fps"], cfg["seconds"],
                                          cfg["ear_thresh"], cfg["blink_window_s"])
            np.save(f, feats)
        X.append(feats); y.append(int(r["Engagement"])); g.append(r["subject"])
    np.savez(cfg["subset_npz"], X=np.stack(X), y=np.array(y), groups=np.array(g))
    print("Saved", cfg["subset_npz"], "| class counts:", np.bincount(y, minlength=4))


def load_splits(cfg):
    """Subject-wise train/val/test split (no subject appears in two splits)."""
    d = np.load(cfg["subset_npz"], allow_pickle=True)
    X, y, g = d["X"], d["y"], d["groups"]
    gss = GroupShuffleSplit(1, test_size=cfg["test_frac"], random_state=cfg["seed"])
    tr, te = next(gss.split(X, y, g))
    gss2 = GroupShuffleSplit(1, test_size=cfg["val_frac"], random_state=cfg["seed"])
    tr2, va = next(gss2.split(X[tr], y[tr], g[tr]))
    tr_idx, va_idx = tr[tr2], tr[va]
    assert not set(g[tr_idx]) & set(g[te]) and not set(g[va_idx]) & set(g[te])
    return (X[tr_idx], y[tr_idx]), (X[va_idx], y[va_idx]), (X[te], y[te])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/config.yaml")
    build_sequences(load_config(ap.parse_args().config))
