"""Clone a voice from a reference clip and speak Vietnamese in it."""
import soundfile as sf

from kokoro_ninja import KokoroNinja

tts = KokoroNinja.load(
    checkpoint_path="checkpoints/kokoro-ninja-multilingual.pth",
    config_path="checkpoints/config_inference.yml",
    campplus_onnx_path="checkpoints/campplus/campplus.onnx",
    device="cuda:0",
)

# Per-voice: compute once, reuse for every sentence in that voice.
style = tts.compute_style("reference.wav")

for i, text in enumerate([
    "Xin chào, đây là giọng nói được nhân bản.",
    "Mỗi cuốn sách là một hành trình kỳ thú.",
]):
    wav, sr = tts.synthesize(text=text, ref_s=style)
    sf.write(f"output_{i}.wav", wav, sr)
    print(f"output_{i}.wav  ({len(wav) / sr:.2f}s)")
