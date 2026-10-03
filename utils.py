import random, yaml, numpy as np, torch

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
