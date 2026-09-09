import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

if "monotonic_align" not in sys.modules:
    from kokoro_ninja import monotonic_align as _stub_ma

    sys.modules["monotonic_align"] = _stub_ma
    sys.modules["monotonic_align.core"] = _stub_ma.core

_STYLETTS2_DIR = os.path.join(_HERE, "StyleTTS2")
if _STYLETTS2_DIR not in sys.path:
    sys.path.insert(0, _STYLETTS2_DIR)

def _ensure_espeak() -> None:
    """Point phonemizer at an espeak-ng library if the system has none.

    phonemizer needs a libespeak-ng shared library and does NOT find the
    pip-installed one on its own, so `pip install espeakng-loader` alone still
    fails with "espeak not installed on your system". Wire it up here so a plain
    pip install is enough. A system espeak, or one the caller configured
    explicitly, always wins.
    """
    try:
        from phonemizer.backend import EspeakBackend
    except Exception:  # phonemizer missing entirely; let the real import fail later
        return
    try:
        if EspeakBackend.is_available():
            return
    except Exception:
        pass
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except Exception:
        pass  # nothing we can do; the backend will raise a clear error itself


_ensure_espeak()

from kokoro_ninja.inference import (  # noqa: E402
    KokoroVI as KokoroNinja,
    FAST_SUPPORTED_LANGUAGES,
)

__all__ = ["KokoroNinja", "FAST_SUPPORTED_LANGUAGES"]
