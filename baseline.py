"""Logistic-regression baseline on per-clip summary statistics."""
import argparse, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .data import load_splits
from .evaluate import report, save
from .utils import load_config


def summarize(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(ap.parse_args().config)
    (Xtr, ytr), _, (Xte, yte) = load_splits(cfg)
    sc = StandardScaler().fit(summarize(Xtr))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(summarize(Xtr)), ytr)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    save(report(yte, clf.predict(sc.transform(summarize(Xte)))),
         os.path.join(cfg["results_dir"], "baseline_metrics.json"))
