from pathlib import Path


def _validate_path(path_str: str, name: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at: {path}")
    return str(path)


def build_models(cfg):
    """
    Build detector, embedder, and optional landmarker based on YAML config.

    Supported runtimes
    ------------------
    "tflite"        TFLite CPU/GPU for detector + embedder + landmarker (CPU)
    "custom"        Alias for "tflite"
    "onnx"          ONNX for detector + embedder (no landmarker)
    "onnx+tflite"   ONNX detector + TFLite embedder + landmarker (CPU)
    "tensorrt"      TensorRT GPU for detector + embedder + landmarker (CPU)
    "tflite+tensorrt" TFLite CPU detector + TensorRT GPU embedder + landmarker (CPU)

    Returns
    -------
    detector, embedder, landmarker (may be None for ONNX-only runtime)
    """

    runtime = cfg["models"].get("runtime", "tflite").lower()
    use_gpu = cfg["models"].get("use_gpu", True)

    detector_path = _validate_path(
        cfg["models"]["detector_path"], "Detector model"
    )

    embedder_path = _validate_path(
        cfg["models"]["embedder_path"], "Embedder model"
    )

    # -------------------------------------------------
    # TFLITE BACKEND  (detector + embedder on CPU/GPU delegate)
    # MediaPipe landmarker is ALWAYS on CPU regardless of use_gpu
    # -------------------------------------------------

    if runtime in ("tflite", "custom"):

        from .detector_tflite import TFLiteUNetDetector
        from .embedder_tflite import TFLiteArcFaceEmbedder
        from .landmarker_tflite import FaceLandmarkerWrapper

        detector = TFLiteUNetDetector(detector_path, use_gpu=use_gpu)
        embedder = TFLiteArcFaceEmbedder(embedder_path, use_gpu=use_gpu)

        landmark_path = _validate_path(
            cfg["models"]["landmark_path"], "Landmarker model"
        )
        # NOTE: FaceLandmarkerWrapper always uses CPU internally; the use_gpu
        # flag is accepted for API consistency but has no effect.
        landmarker = FaceLandmarkerWrapper(landmark_path, use_gpu=use_gpu)

        label = "Custom" if runtime == "custom" else "TFLite"
        print(f"[MODEL_FACTORY] Loaded {label} backend (TFLite stack)")

    # -------------------------------------------------
    # ONNX BACKEND
    # -------------------------------------------------

    elif runtime == "onnx":

        from .detector_onnx import ONNXYuNetDetector
        from .embedder_onnx import ONNXSFaceEmbedder

        detector = ONNXYuNetDetector(detector_path)
        embedder = ONNXSFaceEmbedder(embedder_path)
        landmarker = None

        print("[MODEL_FACTORY] Loaded ONNX backend (no landmarker)")

    # -------------------------------------------------
    # ONNX + TFLITE HYBRID BACKEND
    # -------------------------------------------------

    elif runtime == "onnx+tflite":

        from .detector_onnx import ONNXYuNetDetector
        from .embedder_tflite import TFLiteArcFaceEmbedder
        from .landmarker_tflite import FaceLandmarkerWrapper

        detector = ONNXYuNetDetector(detector_path)
        embedder = TFLiteArcFaceEmbedder(embedder_path, use_gpu=use_gpu)

        landmark_path = _validate_path(
            cfg["models"]["landmark_path"], "Landmarker model"
        )
        landmarker = FaceLandmarkerWrapper(landmark_path, use_gpu=use_gpu)

        print("[MODEL_FACTORY] Loaded ONNX+TFLite hybrid backend")

    # -------------------------------------------------
    # TENSORRT BACKEND  (detector + embedder on GPU via TensorRT)
    # Requires .engine files converted from TFLite/ONNX via trtexec.
    # MediaPipe landmarker still runs on CPU.
    # -------------------------------------------------

    elif runtime == "tensorrt":

        from .detector_tensorrt import TensorRTUNetDetector
        from .embedder_tensorrt import TensorRTArcFaceEmbedder
        from .landmarker_tflite import FaceLandmarkerWrapper

        detector_engine = _validate_path(
            cfg["models"].get("detector_engine_path", detector_path),
            "TensorRT Detector engine"
        )
        embedder_engine = _validate_path(
            cfg["models"].get("embedder_engine_path", embedder_path),
            "TensorRT Embedder engine"
        )

        detector = TensorRTUNetDetector(detector_engine)
        embedder = TensorRTArcFaceEmbedder(embedder_engine)

        landmark_path = _validate_path(
            cfg["models"]["landmark_path"], "Landmarker model"
        )
        landmarker = FaceLandmarkerWrapper(landmark_path, use_gpu=False)

        print("[MODEL_FACTORY] Loaded TensorRT backend — detector + embedder on GPU, landmarker on CPU")

    # -------------------------------------------------
    # TFLITE + TENSORRT HYBRID  (TFLite CPU detector, TRT GPU embedder)
    # Useful when only the ArcFace embedding benefits from GPU.
    # -------------------------------------------------

    elif runtime == "tflite+tensorrt":

        from .detector_tflite import TFLiteUNetDetector
        from .embedder_tensorrt import TensorRTArcFaceEmbedder
        from .landmarker_tflite import FaceLandmarkerWrapper

        embedder_engine = _validate_path(
            cfg["models"].get("embedder_engine_path", embedder_path),
            "TensorRT Embedder engine"
        )

        detector = TFLiteUNetDetector(detector_path, use_gpu=use_gpu)
        embedder = TensorRTArcFaceEmbedder(embedder_engine)

        landmark_path = _validate_path(
            cfg["models"]["landmark_path"], "Landmarker model"
        )
        landmarker = FaceLandmarkerWrapper(landmark_path, use_gpu=False)

        print("[MODEL_FACTORY] Loaded TFLite+TensorRT hybrid — TFLite detector (CPU), TRT embedder (GPU)")

    # -------------------------------------------------
    # UNSUPPORTED BACKEND
    # -------------------------------------------------

    else:
        raise ValueError(
            f"Unsupported runtime type: '{runtime}'. "
            f"Supported values: 'tflite', 'custom', 'onnx', 'onnx+tflite', "
            f"'tensorrt', 'tflite+tensorrt'"
        )

    # -------------------------------------------------
    # FINAL VALIDATION
    # -------------------------------------------------

    if not hasattr(embedder, "embed_dim"):
        raise RuntimeError("Embedder must expose 'embed_dim' property.")

    if embedder.embed_dim <= 0:
        raise RuntimeError(
            f"Invalid embedding dimension detected: {embedder.embed_dim}"
        )

    print(f"[MODEL_FACTORY] Embedding dimension: {embedder.embed_dim}")

    return detector, embedder, landmarker
