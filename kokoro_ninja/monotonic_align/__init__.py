"""Inference-only stub for monotonic_align.

StyleTTS2's utils.py imports `monotonic_align` at module load time but only
invokes it during TRAINING (kokoro-vi-release/CLAUDE.md). This stub satisfies
the import so we can ship without the real `resemble-ai/monotonic_align`
package. If this image is ever reused for training, replace with the real
package — the stub returns zeros and silently breaks alignment.
"""
import torch


def maximum_path(neg_cent, mask):
    return torch.zeros_like(neg_cent, dtype=torch.int32)


def mask_from_lens(lengths, *args, **kwargs):
    max_len = kwargs.get("max_len")
    if max_len is None and args:
        max_len = args[0]
    if max_len is None:
        max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=lengths.device)
    return arange.unsqueeze(0).expand(lengths.shape[0], -1) >= lengths.unsqueeze(1)


class _Core:
    @staticmethod
    def maximum_path_c(*args, **kwargs):
        return None


core = _Core()
