import json, numpy as np
from sklearn.metrics import f1_score, classification_report

LABELS = ["very_low", "low", "high", "very_high"]


def report(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
        "per_class_f1": dict(zip(LABELS, map(float, f1_score(
            y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)))),
    }


def save(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
