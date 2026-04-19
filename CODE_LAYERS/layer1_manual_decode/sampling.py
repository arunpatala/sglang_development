"""
Layer 1 — sampling utilities.

Converts a logit vector (raw model output at one sequence position) into a
single token ID. Kept in its own module so that model.py stays focused on the
decode loop, and so that Layer 2 can reuse or extend this without touching
model.py.
"""

import torch


def sample_next_token(
    logits: torch.Tensor,   # shape: [vocab_size]
    temperature: float,
) -> int:
    """
    Convert a logit vector to a token id.

    temperature=0.0  → greedy (argmax, deterministic)
    temperature=1.0  → multinomial sampling, no scaling
    otherwise        → scale logits by 1/temperature then sample
    """
    if temperature == 0.0:
        return int(logits.argmax(dim=-1).item())

    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
