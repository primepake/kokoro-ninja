import os
import re
import sys
import threading
from collections import OrderedDict

import librosa
import numpy as np
import torch
import torchaudio
import yaml


_HERE = os.path.dirname(os.path.abspath(__file__))
_STYLETTS2_DIR = os.path.join(_HERE, "StyleTTS2")
if _STYLETTS2_DIR not in sys.path:
    sys.path.insert(0, _STYLETTS2_DIR)

from models import build_model, load_ASR_models, load_F0_models  # noqa: E402
from utils import recursive_munch  # noqa: E402
from meldataset import TextCleaner  # noqa: E402
from Utils.PLBERT.util import load_plbert  # noqa: E402

from kokoro_ninja.campplus_extract import CampplusExtractor  # noqa: E402

try:
    from f5.model_interface_zipvoice import _espeak_lock as _ESPEAK_LOCK
except Exception:
    _ESPEAK_LOCK = threading.Lock()


_TARGET_SR = 24000
_CAMPPLUS_SR = 16000


# --- kokoro-vi v4 multilingual phonemization -------------------------------
# One shared model, 7 languages. Vietnamese keeps its custom onset
# disambiguation (kokoro_ninja.phonemize_vi); the other six go through espeak-ng
# with the exact language codes the v4 training set was built with (see the
# release's emilia_index_multi.py LANG_MAP). espeak-ng MUST be 1.50 (the
# Dockerfile pins/verifies it) or the IPA drifts from what the model trained on.
_ESPEAK_LANG = {
    "vi": None,        # custom: kokoro_ninja.phonemize_vi (NOT raw espeak "vi")
    "en": "en-us",
    "zh": "cmn",
    "fr": "fr-fr",
    "de": "de",
    "ja": "ja",
    "ko": "ko",
}
FAST_SUPPORTED_LANGUAGES = tuple(_ESPEAK_LANG.keys())

# espeak emits language-switch tags like "(en)...(vi)" around foreign words plus
# a few stray bracket/tilde chars; TextCleaner would choke on them. Strip both,
# matching the v4 release's clean().
_LANG_TAG_RE = re.compile(r"\([a-z]{2,3}\)")
_STRAY_TABLE = str.maketrans("", "", "(){}̃")


def _clean_phonemes(p):
    return _LANG_TAG_RE.sub("", p).translate(_STRAY_TABLE)


class KokoroVI:
    """Kokoro VI v2 (StyleTTS2 + CAMPPlus speaker embedding) inference wrapper.

    v2 swaps the speaker source from the StyleTTS2 mel style-encoder to CosyVoice2's
    CAMPPlus (192-d) embedding projected into the 256-d style space via ``camp_proj``,
    and drops the diffusion sampler entirely — generation is deterministic.

    Mirrors the surface of the other model interfaces in TTS_Parallel_Inference_Service:
    load once, call ``compute_style(ref_wav_path)`` then ``synthesize(text, ref_s)``.
    The public interface is unchanged from v1; ``synthesize``'s sampler-era kwargs
    (``use_sampler``/``alpha``/``beta``/``diffusion_steps``/``embedding_scale``) are
    accepted for backward compatibility but ignored (v2 has no sampler).
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, checkpoint_path, config_path, campplus_onnx_path, device="cuda:0"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.campplus_onnx_path = campplus_onnx_path
        self._textcleaner = TextCleaner()
        self._espeak_backends = {}  # espeak lang code -> EspeakBackend (lazy, per language)
        self._style_cache = {}
        self._style_cache_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._campplus = CampplusExtractor(campplus_onnx_path)
        self._load_model()

    @classmethod
    def load(cls, checkpoint_path, config_path, campplus_onnx_path, device="cuda:0"):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    checkpoint_path, config_path, campplus_onnx_path, device=device
                )
            return cls._instance

    def _load_model(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        cfg_dir = os.path.dirname(os.path.abspath(self.config_path))

        def _resolve(rel):
            if not rel:
                return rel
            return rel if os.path.isabs(rel) else os.path.join(cfg_dir, rel)

        asr_path = _resolve(config.get("ASR_path", False))
        asr_config = _resolve(config.get("ASR_config", False))
        f0_path = _resolve(config.get("F0_path", False))
        plbert_dir = _resolve(config.get("PLBERT_dir", False))

        text_aligner = load_ASR_models(asr_path, asr_config)
        pitch_extractor = load_F0_models(f0_path)
        plbert = load_plbert(plbert_dir)

        self.model_params = recursive_munch(config["model_params"])
        self.model = build_model(
            self.model_params, text_aligner, pitch_extractor, plbert
        )
        for k in self.model:
            self.model[k].eval()
            self.model[k].to(self.device)

        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        params = state["net"]
        for key in self.model:
            if key not in params:
                continue
            try:
                self.model[key].load_state_dict(params[key])
            except Exception:
                sd = OrderedDict(
                    (k[7:] if k.startswith("module.") else k, v)
                    for k, v in params[key].items()
                )
                self.model[key].load_state_dict(sd, strict=False)
        for k in self.model:
            self.model[k].eval()

    @staticmethod
    def _length_to_mask(lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        return torch.gt(mask + 1, lengths.unsqueeze(1))

    def compute_style(self, ref_wav_path, cache_key=None):
        """Reference wav -> CAMPPlus 192-d -> camp_proj -> 256-d style [acoustic|prosodic].

        Deterministic and robust to out-of-distribution references (CAMPPlus is a
        pretrained speaker-verification model). Returns a (1, 256) tensor on device.
        """
        if cache_key is not None:
            with self._style_cache_lock:
                cached = self._style_cache.get(cache_key)
                if cached is not None:
                    return cached

        wave, _ = librosa.load(ref_wav_path, sr=_CAMPPLUS_SR)
        emb = self._campplus.extract(wave, _CAMPPLUS_SR)  # 192-d
        e = torch.from_numpy(emb).float().unsqueeze(0).to(self.device)
        with self._infer_lock, torch.no_grad():
            out = self.model.camp_proj(e)  # (1, 256)

        if cache_key is not None:
            with self._style_cache_lock:
                if len(self._style_cache) >= 64:
                    self._style_cache.pop(next(iter(self._style_cache)))
                self._style_cache[cache_key] = out
        return out

    def _get_espeak_backend(self, espeak_lang):
        be = self._espeak_backends.get(espeak_lang)
        if be is None:
            from phonemizer.backend import EspeakBackend

            be = EspeakBackend(
                espeak_lang, with_stress=True, preserve_punctuation=True
            )
            self._espeak_backends[espeak_lang] = be
        return be

    def _phonemize(self, text, language="vi"):
        lang = (language or "vi").lower().strip()
        if lang not in _ESPEAK_LANG:
            raise ValueError(
                f"kokoro-vi (fast) supports {list(_ESPEAK_LANG)}; got {language!r}"
            )
        with _ESPEAK_LOCK:
            if lang == "vi":
                # Vietnamese: custom onset disambiguation (tr/s/r), not raw espeak.
                from kokoro_ninja import phonemize_vi as P

                return " ".join(P.phonemize_tokens(text.split()))
            be = self._get_espeak_backend(_ESPEAK_LANG[lang])
            out = be.phonemize([text], strip=True)
            return _clean_phonemes(out[0] if out else "")

    def synthesize(
        self,
        text=None,
        phonemes=None,
        ref_s=None,
        language="vi",
        use_sampler=True,
        alpha=0.3,
        beta=0.7,
        diffusion_steps=5,
        embedding_scale=1.0,
    ):
        # v2 is deterministic: the speaker comes straight from camp_proj(ref_s); there is
        # no diffusion sampler. The sampler-era kwargs above are ignored (kept for the
        # unchanged call surface in tts_service / realtime).
        if ref_s is None:
            raise ValueError("ref_s is required")
        if phonemes is None:
            if not text:
                raise ValueError("Either text or phonemes must be provided")
            phonemes = self._phonemize(text, language)

        with self._infer_lock, torch.no_grad():
            sd = self.model_params.style_dim
            ref = ref_s[:, :sd]   # acoustic -> decoder
            s = ref_s[:, sd:]     # prosodic -> predictor / F0

            tokens = self._textcleaner(phonemes)
            tokens.insert(0, 0)
            tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

            il = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            tmask = self._length_to_mask(il).to(self.device)
            t_en = self.model.text_encoder(tokens, il, tmask)
            bert_dur = self.model.bert(tokens, attention_mask=(~tmask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            d = self.model.predictor.text_encoder(d_en, s, il, tmask)
            x, _ = self.model.predictor.lstm(d)
            dur = torch.sigmoid(self.model.predictor.duration_proj(x)).sum(axis=-1)
            pred_dur = torch.round(dur.squeeze()).clamp(min=1)

            aln = torch.zeros(il, int(pred_dur.sum().data))
            c = 0
            for i in range(aln.size(0)):
                aln[i, c : c + int(pred_dur[i].data)] = 1
                c += int(pred_dur[i].data)

            en = d.transpose(-1, -2) @ aln.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                a = torch.zeros_like(en)
                a[:, :, 0] = en[:, :, 0]
                a[:, :, 1:] = en[:, :, 0:-1]
                en = a
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            asr = t_en @ aln.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                a = torch.zeros_like(asr)
                a[:, :, 0] = asr[:, :, 0]
                a[:, :, 1:] = asr[:, :, 0:-1]
                asr = a
            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        wav = out.squeeze().cpu().numpy()[..., :-50].astype(np.float32)
        wav = np.clip(np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return wav, _TARGET_SR
