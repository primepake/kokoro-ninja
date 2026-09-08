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

from kokoro_ninja.inference import (  # noqa: E402
    KokoroVI as KokoroNinja,
    FAST_SUPPORTED_LANGUAGES,
)

__all__ = ["KokoroNinja", "FAST_SUPPORTED_LANGUAGES"]
