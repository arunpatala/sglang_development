"""
sampler.py — token sampling from a logit distribution.

Mirrors SGLang's sampler (srt/sampling/sampling_utils.py):
  greedy  → argmax (temperature == 0)
  random  → multinomial sample from softmax (temperature > 0)

Extracted here from model_runner._sample so that the sampling policy is
decoupled from KV cache management.  model_runner only calls sample_token();
changing the sampling strategy (top-p, top-k, min-p, repetition penalty)
requires no changes to model_runner or attention logic.
"""

import torch


def sample_token(logits: torch.Tensor, temperature: float) -> int:
    """
    Sample a single token id from `logits`.

    Args:
        logits:      1-D float tensor of shape [vocab_size] (unnormalized).
        temperature: Sampling temperature.
                     0.0  → greedy (argmax, deterministic).
                     >0.0 → multinomial sample from softmax(logits / T).

    Returns:
        Integer token id.
    """
    if temperature == 0.0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))
