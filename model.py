import torch.nn as nn


class EngagementCNN(nn.Module):
    """Lightweight 1D-CNN over (batch, time, features); CPU-friendly (~30k params)."""
    def __init__(self, n_features, n_classes=4):
        super().__init__()
        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.net = nn.Sequential(block(n_features, 32, 5), block(32, 64, 5), block(64, 64, 3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, x):            # x: (B, T, F)
        return self.head(self.net(x.transpose(1, 2)))
