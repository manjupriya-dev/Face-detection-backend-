"""
detector_tensorrt.py
--------------------
TensorRT-backed UNet face detector for Jetson Orin Nano.

Usage
-----
    from .detector_tensorrt import TensorRTUNetDetector
    detector = TensorRTUNetDetector("app/models/unet_detector.engine")
    detections = detector.detect(frame_bgr)

Converting your TFLite model to TensorRT engine
-----------------------------------------------
On the Jetson run:

    /usr/src/tensorrt/bin/trtexec \\
        --uff=unet_model.uff \\
        --output=engine_file.engine

Or via ONNX:
    1. Convert TFLite → ONNX  (e.g. tf2onnx / onnx-tf)
    2. trtexec --onnx=model.onnx --saveEngine=unet_detector.engine

The engine must have:
  - One FP32 input  shape [1, H, W, 3]  (NHWC) or [1, 3, H, W]  (NCHW)
  - One FP32 output shape [1, H', W', 1] or [1, 1, H', W']  (segmentation mask)
"""

from __future__ import annotations

import threading
import numpy as np
import cv2

from .detector_base import Detector
from .detector_tflite import mask_to_bboxes   # reuse post-processing


# ---------------------------------------------------------------------------
# Lazy TensorRT / PyCUDA imports (only needed when this backend is selected)
# ---------------------------------------------------------------------------

def _require_trt():
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  initialises the CUDA context
        return trt, cuda
    except ImportError as exc:
        raise ImportError(
            "TensorRT backend requires 'tensorrt' and 'pycuda'. "
            "In your 'face' conda env run:\n"
            "  pip install pycuda\n"
            "  (tensorrt is normally pre-installed on Jetson via JetPack)"
        ) from exc


# ---------------------------------------------------------------------------
# Helper: allocate host+device buffers for a TensorRT engine
# ---------------------------------------------------------------------------

class _EngineBuffers:
    """Pre-allocated pinned host + device memory for one TRT engine."""

    def __init__(self, engine, cuda):
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.bindings: list[int] = []
        self._cuda = cuda

        for i in range(engine.num_bindings):
            shape = tuple(engine.get_binding_shape(i))
            size = int(np.prod(shape))
            dtype = np.float32  # engines are expected to use FP32

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            entry = {"host": host_mem, "device": device_mem, "shape": shape}
            if engine.binding_is_input(i):
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)


# ---------------------------------------------------------------------------
# TensorRT UNet Detector
# ---------------------------------------------------------------------------

class TensorRTUNetDetector(Detector):
    """
    Drop-in replacement for TFLiteUNetDetector using a TensorRT .engine file.
    Shares the same mask_to_bboxes post-processing as the TFLite version.
    """

    def __init__(
        self,
        engine_path: str,
        rgb: bool = True,
        mask_thresh: float = 0.5,
        conf_thresh: float = 0.3,
        min_area_px: int = 1500,
        max_faces: int = 5,
    ):
        trt, cuda = _require_trt()

        self._cuda = cuda
        self._lock = threading.Lock()

        # ---- Load serialised engine ----
        logger_trt = trt.Logger(trt.Logger.WARNING)
        runtime_trt = trt.Runtime(logger_trt)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self._engine = runtime_trt.deserialize_cuda_engine(engine_data)
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        self._bufs = _EngineBuffers(self._engine, cuda)

        # ---- Detect input layout ----
        inp_shape = self._bufs.inputs[0]["shape"]   # e.g. (1, 416, 416, 3) or (1, 3, 416, 416)
        if inp_shape[1] == 3:
            self._layout = "NCHW"
            self._input_size = (inp_shape[2], inp_shape[3])   # (W, H)
        else:
            self._layout = "NHWC"
            self._input_size = (inp_shape[1], inp_shape[2])   # (W, H) — actually H, W but kept symmetric

        self.rgb = rgb
        self.mask_thresh = mask_thresh
        self.conf_thresh = conf_thresh
        self.min_area_px = min_area_px
        self.max_faces = max_faces

        self._last_scale = 1.0
        self._last_x_offset = 0
        self._last_y_offset = 0

        print(f"[TRT DETECTOR] Engine loaded | input {inp_shape} | backend: GPU (TensorRT)")

    # ------------------------------------------------------------------
    # Preprocess  (same letterbox logic as TFLite version)
    # ------------------------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if self.rgb else frame_bgr

        target_w, target_h = self._input_size
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        self._last_scale = scale
        self._last_x_offset = x_offset
        self._last_y_offset = y_offset

        x = canvas.astype(np.float32) / 255.0

        if self._layout == "NCHW":
            x = np.transpose(x, (2, 0, 1))

        return np.expand_dims(x, 0).astype(np.float32)

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            return []

        orig_h, orig_w = frame_bgr.shape[:2]
        x = self._preprocess(frame_bgr)

        with self._lock:
            inp_buf = self._bufs.inputs[0]
            out_buf = self._bufs.outputs[0]

            np.copyto(inp_buf["host"], x.ravel())
            self._cuda.memcpy_htod_async(inp_buf["device"], inp_buf["host"], self._stream)
            self._context.execute_async_v2(self._bufs.bindings, self._stream.handle)
            self._cuda.memcpy_dtoh_async(out_buf["host"], out_buf["device"], self._stream)
            self._stream.synchronize()

            out = out_buf["host"].reshape(out_buf["shape"]).copy()

        # Squeeze to 2-D mask
        mask_small = np.squeeze(out).astype(np.float32)

        dets = mask_to_bboxes(
            mask_small,
            orig_size=(orig_w, orig_h),
            pad_info=(self._last_scale, self._last_x_offset, self._last_y_offset),
            input_size=self._input_size,
            thresh=self.mask_thresh,
            min_area_px=self.min_area_px,
            max_faces=self.max_faces,
        )

        dets = [d for d in dets if d["score"] >= self.conf_thresh]
        return self._validate_detections(frame_bgr, dets)
