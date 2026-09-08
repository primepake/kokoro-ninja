"""CAMPPlus speaker-embedding extractor (kokoro-vi v2) — exact CosyVoice2 recipe.

audio -> 16 kHz mono -> kaldi.fbank(80, dither=0) -> per-utterance mean-norm ->
campplus.onnx -> 192-d float32 speaker embedding.

Unlike the v1 mel style-encoder, this runs on CPU via onnxruntime (campplus.onnx is
~28 MB and cheap), so it does not contend for the GPU. The ONNX path is injected at
construction time (no hardcoded paths) so it resolves from the deployed checkpoint dir.
"""

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


class CampplusExtractor:
    def __init__(self, onnx_path):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        # CPU is intentional: the speaker model is tiny and we keep the GPU for synthesis.
        self._sess = onnxruntime.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name

    def extract(self, wav, sr):
        """wav: 1-D float np array or torch tensor. Returns a 192-d float32 np array."""
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self._sess.run(
            None, {self._input_name: feat.unsqueeze(0).cpu().numpy()}
        )[0].flatten()
        return emb.astype(np.float32)
