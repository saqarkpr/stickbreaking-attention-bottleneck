"""
Synthetic sequence-classification task designed to test whether a
bottleneck that can *compress away irrelevant tokens* generalizes better
under distribution shift than a model that pools everything uniformly.

Vocabulary:
  - tokens 0..NOISE_VOCAB-1        : noise tokens, carry no label information
  - tokens NOISE_VOCAB..+N_CLASSES : "signal" tokens, one per class

A sequence's label is the class of whichever signal token appears in it
(each sequence contains exactly one signal token, placed at a random
position, surrounded by noise). This means the useful information is a
single token buried in a sequence of distractors — exactly the setting
where a sparse, compressive bottleneck should have an advantage over
uniform mean-pooling, and the advantage should widen as the noise ratio
(sequence length) increases beyond what was seen in training.
"""
import torch

NOISE_VOCAB = 10
N_CLASSES = 4
SIGNAL_TOKENS = list(range(NOISE_VOCAB, NOISE_VOCAB + N_CLASSES))
VOCAB_SIZE = NOISE_VOCAB + N_CLASSES
PAD_TOKEN = -1  # handled via pad_mask, not embedded


def make_batch(batch_size: int, seq_len: int, device: str = "cpu"):
    labels = torch.randint(0, N_CLASSES, (batch_size,))
    signal_tok = torch.tensor(SIGNAL_TOKENS, device=device)[labels]

    x = torch.randint(0, NOISE_VOCAB, (batch_size, seq_len), device=device)
    signal_pos = torch.randint(0, seq_len, (batch_size,))
    x[torch.arange(batch_size), signal_pos] = signal_tok

    pad_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    return x, labels.to(device), pad_mask


if __name__ == "__main__":
    x, y, mask = make_batch(4, 12)
    print("x:\n", x)
    print("y:", y)
    print("signal tokens are in [10..13], noise tokens in [0..9]")
