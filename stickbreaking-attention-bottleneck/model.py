import torch
import torch.nn as nn

from encoder import TransformerEncoder
from bottleneck_layer import StickBreakingBottleneck


class BaselineClassifier(nn.Module):
    """Transformer encoder + uniform mean-pooling + linear head."""

    def __init__(self, vocab_size, n_classes, d_model=64, n_heads=4, n_layers=3, max_len=128):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, n_heads, n_layers, max_len)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, idx, pad_mask=None):
        h = self.encoder(idx, pad_mask)
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)
        logits = self.head(pooled)
        return logits, torch.tensor(0.0, device=idx.device)  # no KL term, kept for interface parity


class BottleneckClassifier(nn.Module):
    """Transformer encoder + stick-breaking variational bottleneck pooling + linear head."""

    def __init__(self, vocab_size, n_classes, d_model=64, n_heads=4, n_layers=3, max_len=128, alpha0=5.0, shuffle_sticks=False):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, n_heads, n_layers, max_len)
        self.bottleneck = StickBreakingBottleneck(d_model, alpha0=alpha0, shuffle_sticks=shuffle_sticks)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, idx, pad_mask=None):
        h = self.encoder(idx, pad_mask)
        pooled, kl, pi = self.bottleneck(h, pad_mask)
        logits = self.head(pooled)
        return logits, kl


if __name__ == "__main__":
    from data import make_batch, VOCAB_SIZE, N_CLASSES

    x, y, mask = make_batch(4, 16)
    base = BaselineClassifier(VOCAB_SIZE, N_CLASSES)
    bottleneck = BottleneckClassifier(VOCAB_SIZE, N_CLASSES)

    logits_b, kl_b = base(x, mask)
    logits_n, kl_n = bottleneck(x, mask)
    print("baseline logits:", logits_b.shape, "kl:", kl_b.item())
    print("bottleneck logits:", logits_n.shape, "kl:", kl_n.item())
