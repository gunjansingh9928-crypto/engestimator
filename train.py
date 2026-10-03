"""Train the 1D-CNN with class-weighted loss; evaluate on held-out subjects."""
import argparse, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .data import load_splits
from .evaluate import report, save
from .model import EngagementCNN
from .utils import load_config, set_seed


def normalize(tr, *others):
    mu, sd = tr.mean((0, 1), keepdims=True), tr.std((0, 1), keepdims=True) + 1e-6
    return [(a - mu) / sd for a in (tr, *others)], (mu, sd)


def train_cnn(cfg, keep=None, save_path=None):
    """keep: optional list of feature indices (used by the ablation)."""
    set_seed(cfg["seed"])
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_splits(cfg)
    if keep is not None:
        Xtr, Xva, Xte = Xtr[..., keep], Xva[..., keep], Xte[..., keep]
    (Xtr, Xva, Xte), (mu, sd) = normalize(Xtr, Xva, Xte)

    counts = np.bincount(ytr, minlength=4).astype(float)
    w = torch.tensor(counts.sum() / (4 * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    model = EngagementCNN(Xtr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    loader = DataLoader(TensorDataset(t(Xtr), t(ytr, torch.long)),
                        batch_size=cfg["batch_size"], shuffle=True)

    def predict(X):
        model.eval()
        with torch.no_grad():
            return model(t(X)).argmax(1).numpy()

    best, best_state = -1, None
    for ep in range(cfg["epochs"]):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        f1 = report(yva, predict(Xva))["macro_f1"]
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if save_path:
        torch.save({"state": best_state, "mu": mu, "sd": sd}, save_path)
    return report(yte, predict(Xte))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(ap.parse_args().config)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    save(train_cnn(cfg, save_path=os.path.join(cfg["results_dir"], "cnn.pt")),
         os.path.join(cfg["results_dir"], "cnn_metrics.json"))
