"""Small bidirectional Transformer encoder (no causal mask) — the backbone
that the stick-breaking bottleneck layer sits on top of."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if pad_mask is not None:
            # pad_mask: (B, T) True where token is real, False where padding
            mask = pad_mask[:, None, None, :]
            att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=3, max_len=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, idx, pad_mask=None):
        x = self.pos(self.emb(idx))
        for b in self.blocks:
            x = b(x, pad_mask)
        return self.ln_f(x)  # (B, T, D) per-token hidden states
