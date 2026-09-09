# Kokoro Ninja 🥷

**Fast multilingual text-to-speech with zero-shot voice cloning.** Give it a few seconds of
someone's voice and some text, and it speaks in that voice — at **~15× realtime on a single
GPU**.

**7 languages:** Vietnamese, English, Chinese, French, German, Japanese, Korean — one model,
one speaker embedding, so a cloned voice carries across every language it supports.

**[🎧 Listen to the demo](https://primepake.github.io/kokoro-ninja/)** — side-by-side against
VieNeu-TTS and omnivoice-vietnamese on held-out speakers, plus one voice across all seven
languages. **[Weights on HuggingFace](https://huggingface.co/prime2070/kokoro-ninja)**.

[StyleTTS2](https://github.com/yl4579/StyleTTS2) + a
[CAMPPlus](https://github.com/modelscope/3D-Speaker) speaker encoder. One non-autoregressive
forward pass: no diffusion sampler, no autoregressive decode loop. That's where the speed comes
from, and it makes generation deterministic — same input, same output.

## Install

```bash
git clone https://github.com/primepake/kokoro-ninja
cd kokoro-ninja
pip install -r requirements.txt
```

**espeak-ng** is the phonemizer backend, and `requirements.txt` already pulls in
`espeakng-loader` (which ships espeak-ng 1.52 with complete dictionaries). `kokoro_ninja` wires
it into phonemizer automatically on import, so there is nothing else to do. A system espeak, if
you have one, takes precedence.

The model was trained against espeak-ng **1.50**. We compared 1.50 and 1.52 through this repo's
phonemizer and got **identical phonemes on all 30 Vietnamese benchmark sentences**, plus identical
output for English, French, German, Japanese and Korean — so 1.52 is fine. Only Mandarin differs,
where 1.52 is the better of the two (1.50 ships no `cmn_dict`).

If you do build 1.50 from source, **compile the dictionaries explicitly**. A plain `make` can leave
them corrupt, and a broken `en_dict` makes espeak spell words out letter by letter
("B-O-N-J-O-U-R") — which silently wrecks English, French and German while Vietnamese still sounds
fine, so it is easy to miss:

```bash
cd espeak-ng/dictsource
for L in en-us fr de ja ko vi; do
  ESPEAK_DATA_PATH=../espeak-ng-data LD_LIBRARY_PATH=../src/.libs ../src/espeak-ng --compile=$L
done
```

Then put the weights bundle in `checkpoints/`. `config_inference.yml` resolves the asset paths
relative to itself, so keep it beside them:

```
checkpoints/
├── config_inference.yml
├── kokoro-ninja-multilingual.pth
├── campplus/campplus.onnx
└── Utils/{ASR,JDC,PLBERT}/...
```

## Usage

```python
import soundfile as sf
from kokoro_ninja import KokoroNinja

tts = KokoroNinja.load(
    checkpoint_path="checkpoints/kokoro-ninja-multilingual.pth",
    config_path="checkpoints/config_inference.yml",
    campplus_onnx_path="checkpoints/campplus/campplus.onnx",
    device="cuda:0",
)

style = tts.compute_style("reference.wav")        # 3-10s of the voice to clone

wav, sr = tts.synthesize(
    text="Xin chào, đây là giọng nói được nhân bản.",
    ref_s=style,
    language="vi",
)
sf.write("output.wav", wav, sr)
```

`compute_style()` is per-voice — compute it once and reuse it for every sentence in that voice.

The same speaker embedding works across languages, so one reference clip clones a voice into
any of the seven:

```python
from kokoro_ninja import FAST_SUPPORTED_LANGUAGES
print(FAST_SUPPORTED_LANGUAGES)   # ('vi', 'en', 'zh', 'fr', 'de', 'ja', 'ko')

for lang, text in [
    ("en", "Hello, this is a cloned voice."),
    ("ja", "こんにちは、これはクローンされた声です。"),
    ("fr", "Bonjour, ceci est une voix clonée."),
]:
    wav, sr = tts.synthesize(text=text, ref_s=style, language=lang)
    sf.write(f"output_{lang}.wav", wav, sr)
```

## Benchmarks

Measured on one NVIDIA L4, VIVOS test split, 10 held-out speakers × 3 sentences, same
references and sentences for every system:

| System | RTF ↓ | ×realtime | SECS ↑ | UTMOSv2 ↑ | WER ↓ |
|---|---|---|---|---|---|
| **Kokoro Ninja** | **0.061** | **16.3×** | 0.894 | 2.660 | 3.9% |
| omnivoice-vietnamese | 0.437 | 2.3× | 0.916 | 2.660 | 2.6% |
| VieNeu-TTS v3 Turbo | 0.718 | 1.4× | 0.931 | 2.445 | 3.3% |
| *human recording* | — | — | *0.914* | *3.019* | *5.7%* |

Honestly: **we win on speed by 7–12×** and tie omnivoice-vietnamese for the best predicted
naturalness, but **speaker similarity trails** the autoregressive models — 0.894 against a 0.833
floor (two different speakers) and a 0.914 ceiling (the same speaker's other recordings), so we
reach about three-quarters of the usable range while they reach it or pass it. If you need
maximum timbre fidelity from one clip rather than throughput, VieNeu v3 Turbo is currently
better. Intelligibility is below the human recordings' own 5.7% WER.

RTF is full-utterance compute ÷ audio duration, not time-to-first-audio. SECS is scored with
WavLM-base-plus-sv rather than CAMPPlus, since scoring with the encoder the model is
conditioned on would flatter it. WER is PhoWhisper back-transcription, which catches wrong
tones because a wrong tone is a different Vietnamese word. UTMOSv2 is also nondeterministic —
rescoring identical audio moves it by up to 0.08, so treat small MOS gaps as noise; SECS and WER
are exactly reproducible.

### Always end your text with punctuation

The model is trained on sentence-final punctuation and is genuinely sensitive to it — text
without a terminal mark tends to clip or run on at the end. `synthesize()` appends one when your
text lacks it. The effect is not subtle: on the benchmark above, adding the missing period moved
WER from 5.1% to 3.9% and predicted MOS from 2.47 to 2.66.

## Vietnamese phonemization

espeak-ng's `vi` voice collapses phonemically distinct onsets into one IPA token, which makes
them unlearnable. `kokoro_ninja/phonemize_vi.py` fixes the three that matter:

| written | espeak gives | should be |
|---|---|---|
| `tr` vs `ch` | both `tʃ` | `ʈ` vs `tʃ` |
| `s` vs `x` | both `s` | `ʂ` vs `s` |
| `r` vs `d`/`gi` | all `z` | `ʐ` vs `z` |

It phonemizes per syllable, looks at the written onset and remaps the colliding phoneme to the
right retroflex — already in the vocabulary, so no vocab change. Guarded: it only fires when
espeak actually emitted the colliding token. (`c`/`k`/`q` → `k` is left alone — a genuine
merger in Vietnamese.)

## Limitations

- **Speaker similarity** trails the best autoregressive/codec models (see above).
- **Benchmarked on Vietnamese only** — the other six languages work but are not yet measured.
- **Japanese input must be kana.** espeak-ng reads hiragana and katakana correctly but
  silently *drops* kanji, so run text through a kanji→kana converter (pyopenjtalk, MeCab)
  first. Every other language takes normal text.
- **No text normalization.** Expand numbers, dates and currency before synthesis — this is the
  most common source of real-world errors in Vietnamese TTS and it is not handled here.
- Noisy or very short (<3s) references degrade cloning; there is no built-in denoiser.
- 24 kHz output, no streaming.

## Training data

| Corpus | Role | Licence |
|---|---|---|
| [viVoice](https://huggingface.co/datasets/capleaf/viVoice) | Vietnamese, ~1,000 h | **CC-BY-NC-SA-4.0** |
| [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) (subset) | multilingual | **CC-BY-NC-4.0** |
| internal corpus | Vietnamese | proprietary |

Base checkpoint: StyleTTS2's LibriTTS model, adapted to the Vietnamese phoneme vocabulary and
then extended to the multilingual token set.

## Licence

**Code: Apache-2.0** ([LICENSE](LICENSE)) — this repository's source.

**Weights: CC-BY-NC-SA-4.0 — non-commercial.** This is not a choice, it is inherited: viVoice
is CC-BY-NC-SA (NonCommercial *and* ShareAlike) and the Emilia subset is CC-BY-NC, so the
strictest terms propagate to anything trained on them. Concretely, you may use the weights for
research, evaluation and personal projects; you may **not** use them in a commercial product,
and derivatives must carry the same licence. If you need commercial terms, you need a model
trained without those corpora.

Built on [StyleTTS2](https://github.com/yl4579/StyleTTS2) (MIT),
[CAMPPlus / 3D-Speaker](https://github.com/modelscope/3D-Speaker) (Apache-2.0), and
[espeak-ng](https://github.com/espeak-ng/espeak-ng) (GPL-3.0, used as a separate runtime
library).
