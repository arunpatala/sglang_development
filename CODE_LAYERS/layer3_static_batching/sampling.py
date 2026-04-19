"""
Layer 3 — sampling utilities.

Layers 1 and 2 exposed sample_next_token, which operates on a single
[vocab_size] logit vector and returns a scalar int. Layer 3 processes B
requests simultaneously, so sampling operates on a [B, vocab_size] matrix
and returns a [B] tensor of token ids.

Kept in each layer so that every layer is self-contained and can be run
independently.
"""

import torch


def sample_batch(
    logits: torch.Tensor,   # shape: [B, vocab_size]
    temperature: float,
) -> torch.Tensor:          # shape: [B]
    """
    Convert a batch of logit vectors to a batch of token ids.

    temperature=0.0  → greedy (argmax per row, deterministic)
    temperature=1.0  → multinomial sampling, no scaling
    otherwise        → scale logits by 1/temperature then sample
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
