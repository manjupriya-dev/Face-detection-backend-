"""
gpu_delegate.py
---------------
Shared GPU delegate detection for TFLite models (UNet detector, ArcFace embedder).

NOTE: MediaPipe FaceLandmarker is NOT handled here — it always runs on CPU because
the mediapipe build on Jetson Orin has GPU processing disabled at the build level.

Strategy
--------
1. Detect the platform (Jetson vs CUDA desktop vs other).
2. On Jetson, probe the TFLite OpenGL ES delegate (libGLESv2 / EGL — available on
   Jetson Orin via the NVIDIA GPU driver). NNAPI is NOT used because libnnapi.so is
   absent on Jetson Orin Nano Super.
3. On a CUDA desktop, probe the OpenGL ES delegate as before.
4. Fall back to CPU silently if no delegate loads.

Probe order:
  Jetson Orin  → OpenGL ES delegate  (libGLESv2.so / libEGL.so)
  CUDA desktop → OpenGL ES delegate  (libgles.so / libflutter_embedding.so)
  Everything else → CPU

Usage
-----
    from .gpu_delegate import make_delegate

    delegates, backend = make_delegate(use_gpu=True)
    interp = tflite.Interpreter(
        model_path=path,
        experimental_delegates=delegates,
        num_threads=num_threads,
    )
    print(f"[MODEL] Running on {backend}")
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware-presence checks  (all instant – no library loading)
# ---------------------------------------------------------------------------

def _is_jetson() -> bool:
    """True when running on an NVIDIA Jetson board (Nano, Xavier, Orin…)."""
    if Path("/etc/nv_tegra_release").exists():
        return True
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        try:
            return b"NVIDIA" in model_file.read_bytes()
        except OSError:
            pass
    return False


def _has_nvidia_gpu() -> bool:
    """True when an NVIDIA GPU is present (CUDA desktop / workstation)."""
    if Path("/dev/nvidia0").exists():
        return True
    if Path("/proc/driver/nvidia/version").exists():
        return True
    return False


def _find_lib(name: str) -> str | None:
    """
    Return the found name if the library exists on LD_LIBRARY_PATH or standard
    linker search paths without actually loading it. Returns None if not found.
    """
    import ctypes.util
    return ctypes.util.find_library(name.replace("lib", "", 1).replace(".so", ""))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_delegate(use_gpu: bool = True) -> tuple[list, str]:
    """
    Try to build the best available TFLite GPU delegate.

    Parameters
    ----------
    use_gpu : bool
        When False, skip all GPU probing and return CPU immediately.

    Returns
    -------
    delegates : list
        Pass as ``experimental_delegates`` to tflite.Interpreter.
        Empty list → CPU-only (tflite_runtime default).
    backend : str
        Human-readable label: "GPU (OpenGL ES / libXxx.so)", or "CPU".
    """

    if not use_gpu:
        return [], "CPU (forced)"

    try:
        from tflite_runtime.interpreter import load_delegate
    except ImportError:
        print("[GPU DELEGATE] tflite_runtime has no delegate support — using CPU")
        return [], "CPU"

    # ------------------------------------------------------------------
    # Jetson path: OpenGL ES / EGL delegate (NNAPI absent on Orin Nano Super)
    # ------------------------------------------------------------------
    if _is_jetson():
        jetson_libs = [
            "libGLESv2.so",      # Most common on Orin
            "libEGL.so",         # Alternative EGL path
            "libGLESv2.so.2",    # Versioned symlink
        ]
        for lib in jetson_libs:
            lib_name = lib.replace(".so.2", "").replace(".so", "").replace("lib", "", 1)
            if _find_lib(lib_name) is None:
                continue
            try:
                delegate = load_delegate(lib)
                print(f"[GPU DELEGATE] Jetson detected — using GPU delegate ({lib})")
                return [delegate], f"GPU (OpenGL ES / {lib})"
            except Exception as exc:
                logger.debug("[GPU DELEGATE] %s load failed on Jetson: %s", lib, exc)

        print(
            "[GPU DELEGATE] Jetson detected but no TFLite GPU delegate found — using CPU.\n"
            "               (Set  models.use_gpu: false  in config.yaml to suppress this message)\n"
            "               Tip: use  runtime: \"tensorrt\"  for GPU acceleration on Jetson."
        )
        return [], "CPU"

    # ------------------------------------------------------------------
    # CUDA desktop path: try OpenGL ES delegate
    # ------------------------------------------------------------------
    if _has_nvidia_gpu():
        for lib in ("libgles.so", "libflutter_embedding.so"):
            if _find_lib(lib) is None:
                continue
            try:
                delegate = load_delegate(lib)
                print(f"[GPU DELEGATE] NVIDIA GPU detected — using GPU delegate: OpenGL ES ({lib})")
                return [delegate], f"GPU (OpenGL ES / {lib})"
            except Exception as exc:
                logger.debug("[GPU DELEGATE] %s load failed: %s", lib, exc)

    # ------------------------------------------------------------------
    # CPU fallback
    # ------------------------------------------------------------------
    logger.info(
        "[GPU DELEGATE] No supported GPU hardware detected — using CPU. "
        "To disable this probe, set  models.use_gpu: false  in config.yaml."
    )
    print(
        "[GPU DELEGATE] No supported GPU hardware detected — using CPU.\n"
        "               (Set  models.use_gpu: false  in config.yaml to suppress this message)"
    )
    return [], "CPU"


# ---------------------------------------------------------------------------
# TensorRT availability check  (used by model_factory for early validation)
# ---------------------------------------------------------------------------

def check_tensorrt_available() -> bool:
    """Return True if both tensorrt and pycuda are importable."""
    try:
        import tensorrt  # noqa: F401
        import pycuda.driver  # noqa: F401
        return True
    except ImportError:
        return False
