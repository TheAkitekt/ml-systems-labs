import torch
import torch.nn as nn


class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FashionAttention(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, dim_feedforward: int = 128):
        super().__init__()
        self.patch_proj = nn.Linear(49, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 28, 28) -> 16 patches of 7x7 -> (B, 16, 49)
        b = x.shape[0]
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7).contiguous().view(b, 16, 49)
        tokens = self.patch_proj(patches)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)
