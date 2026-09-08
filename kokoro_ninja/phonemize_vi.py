#!/usr/bin/env python
"""Vietnamese phonemization with ONSET DISAMBIGUATION for StyleTTS2.

espeak-ng `vi` collapses phonemically-distinct Vietnamese syllable onsets to one
IPA token, which makes them unlearnable by the TTS:
    tr & ch  -> tʃ     |  s & x -> s     |  r & d/gi -> z
(c/k/q -> k is a GENUINE merger in Vietnamese — left alone.)

Fix: phonemize per-syllable (Vietnamese is whitespace-delimited & isolating, so
word≈syllable), look at the orthographic onset, and remap the colliding onset
phoneme to the correct retroflex — which is ALREADY in the StyleTTS2 187 vocab,
so no vocab change and the warm-started embeddings already exist:
    tr -> ʈ   (ch keeps tʃ)
    s  -> ʂ   (x  keeps s)
    r  -> ʐ   (d, gi keep z)
The remap is GUARDED: it only fires when the espeak output actually starts with
the expected colliding token, otherwise the phonemes are left untouched.

Modes:
  --test                  print the disambiguation on known minimal pairs
  --build OUT_PREFIX      read vi_manifests/{vieneu,dolly,vivoice}.jsonl and write
                          combined StyleTTS2 manifests (abs_path|phonemes|spk).
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor

# espeak emits language-switch tags like "(en)...(vi)" around foreign words, plus
# occasional stray braces and a rare combining tilde — none are in the vocab and
# TextCleaner would silently corrupt the stream (keeping the inner 'en'/'vi'
# letters). Strip the TAGS (keep the foreign phonemes between them) + stray chars.
_LANG_TAG = re.compile(r"\([a-z]{2,3}\)")
_STRAY = str.maketrans("", "", "(){}̃")

def _clean(p):
    return _LANG_TAG.sub("", p).translate(_STRAY)

VI_MANIFESTS = os.environ.get("VI_MANIFESTS", "data/vi_manifests")
SOURCES = ["vieneu", "dolly", "vivoice"]
MAX_PHON = 508                      # ALBERT 512 limit (minus BOS/EOS margin)
LEAD_PUNCT = "““«»”‘’'\"()[]¿¡.,;:!?—…- \t"

_backend = None
def _be():
    global _backend
    if _backend is None:
        from phonemizer.backend import EspeakBackend
        _backend = EspeakBackend("vi", with_stress=True, preserve_punctuation=True)
    return _backend


def _onset(word):
    w = word.lower().lstrip(LEAD_PUNCT)
    if w.startswith("tr"):
        return "tr"
    if w.startswith("s"):
        return "s"
    if w.startswith("r"):
        return "r"
    return None


def _disambig(word, phon):
    """Remap the leading onset phoneme based on the orthographic onset. Guarded."""
    o = _onset(word)
    if o == "tr" and phon.startswith("tʃ"):   # tʃ -> ʈ
        return "ʈ" + phon[2:]
    if o == "s" and phon.startswith("s"):           # s -> ʂ
        return "ʂ" + phon[1:]
    if o == "r" and phon.startswith("z"):           # z -> ʐ
        return "ʐ" + phon[1:]
    return phon


def phonemize_tokens(tokens):
    """tokens: list[str] -> list[str] disambiguated phonemes (per token)."""
    raw = _be().phonemize(tokens)
    return [_clean(_disambig(t, p.strip())) for t, p in zip(tokens, raw)]


def phonemize_text(text, cache):
    """Reconstruct a clip's phoneme string from the per-token cache, join on space."""
    return " ".join(cache[t] for t in text.split() if t in cache)


def _run_test():
    pairs = [("tra", "cha"), ("trắng", "chăng"), ("trâu", "châu"),
             ("sa", "xa"), ("sạch", "xạch"), ("ra", "da"), ("ra", "gia")]
    flat = [w for p in pairs for w in p]
    ph = dict(zip(flat, phonemize_tokens(flat)))
    print("orthographic pair -> disambiguated IPA (must be DISTINCT):")
    for a, b in pairs:
        print(f"  {a:8s} -> {ph[a]!r:12}   {b:8s} -> {ph[b]!r:12}   distinct={ph[a] != ph[b]}")


def _phon_chunk(tokens):
    return phonemize_tokens(tokens)


def _build(out_prefix, workers):
    t0 = time.time()
    # 1) load all clips, collect unique tokens
    clips = []          # (abs_path, text, source)
    uniq = set()
    for src in SOURCES:
        n = 0
        for line in open(os.path.join(VI_MANIFESTS, f"{src}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            clips.append((r["audio_path"], r["text"], src))
            uniq.update(r["text"].split())
            n += 1
        print(f"[load] {src}: {n} clips", flush=True)
    uniq = sorted(uniq)
    print(f"[load] {len(clips):,} clips, {len(uniq):,} unique tokens, {time.time()-t0:.0f}s", flush=True)

    # 2) phonemize unique tokens in parallel chunks
    CH = 2000
    chunks = [uniq[i:i + CH] for i in range(0, len(uniq), CH)]
    cache = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for chunk, res in zip(chunks, ex.map(_phon_chunk, chunks)):
            cache.update(zip(chunk, res))
            done += len(chunk)
            if done % 20000 < CH:
                print(f"[phon] {done:,}/{len(uniq):,} tokens ({done/(time.time()-t0+1):.0f}/s)", flush=True)
    print(f"[phon] done {len(cache):,} tokens, {time.time()-t0:.0f}s", flush=True)

    # 3) vocab coverage check
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "StyleTTS2"))
    from meldataset import dicts
    unknown = {}
    for p in cache.values():
        for ch in p:
            if ch not in dicts:
                unknown[ch] = unknown.get(ch, 0) + 1
    if unknown:
        print(f"[vocab] WARNING unknown symbols: {sorted(unknown.items(), key=lambda x:-x[1])[:20]}", flush=True)
    else:
        print("[vocab] OK: all phoneme symbols are in the 187 vocab", flush=True)

    # 4) write manifests (abs_path|phonemes|unique_spk), filter long.
    #    NOTE: no per-file os.path.exists (1.6M NFS stats ~= 2h). Files come from a
    #    completed extraction; we do a SAMPLED integrity check below instead.
    spk = 0
    dropped_long = 0
    val, ood = [], []
    fout = open(f"{out_prefix}.txt", "w", encoding="utf-8")
    for i, (path, text, src) in enumerate(clips):
        ph = phonemize_text(text, cache)
        if len(ph) > MAX_PHON:
            dropped_long += 1
            continue
        line = f"{path}|{ph}|{spk}"
        fout.write(line + "\n")
        if i % 8000 == 0 and len(val) < 200:
            val.append(line)
        if i % 200 == 0 and len(ood) < 8000:
            ood.append(line)
        spk += 1
    fout.close()
    open(f"{os.path.dirname(out_prefix)}/val_vi.txt", "w", encoding="utf-8").write("\n".join(val) + "\n")
    open(f"{os.path.dirname(out_prefix)}/OOD_texts_vi.txt", "w", encoding="utf-8").write("\n".join(ood) + "\n")
    print(f"[build] wrote {spk:,} lines -> {out_prefix}.txt | dropped long={dropped_long}", flush=True)

    # sampled integrity check: 2000 random paths must exist
    import random as _r
    lines = open(f"{out_prefix}.txt", encoding="utf-8").read().splitlines()
    sample = _r.Random(0).sample(lines, min(2000, len(lines)))
    miss = sum(1 for ln in sample if not os.path.exists(ln.split("|")[0]))
    print(f"[verify] sampled {len(sample)} paths: {miss} missing ({100*miss/len(sample):.2f}%)", flush=True)
    print(f"[build] val={len(val)} ood={len(ood)} | total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--build", metavar="OUT_PREFIX", default=None)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if a.test:
        _run_test()
    if a.build:
        _build(a.build, a.workers)
