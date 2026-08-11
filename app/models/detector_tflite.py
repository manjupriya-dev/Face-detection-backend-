import numpy as np
import cv2
import threading
from tflite_runtime.interpreter import Interpreter

from .detector_base import Detector
from .gpu_delegate import make_delegate


# -------------------------------------------------
# UTILS
# -------------------------------------------------

def _dequantize(x, detail):
    scale, zero = detail.get("quantization", (0.0, 0))
    if scale and scale > 0:
        return (x.astype(np.float32) - zero) * scale
    return x.astype(np.float32)


def _get_2d_mask(out: np.ndarray) -> np.ndarray:
    return np.squeeze(out).astype(np.float32)


def mask_to_bboxes(
    mask: np.ndarray,
    orig_size: tuple,  # (W, H)
    pad_info: tuple, # (scale, x_offset, y_offset)
    input_size: tuple, # (unet_w, unet_h)
    thresh: float = 0.5,
    min_area_px: int = 1500,
    max_faces: int = 5,
    morph_open: int = 3,
    morph_close: int = 7,
):
    """Process mask at its native resolution and scale bboxes to original size."""
    mask_h, mask_w = mask.shape
    orig_w, orig_h = orig_size

    # Not used anymore since we handle scaling via padding info.
    # We will pass scale and offsets to this function directly.

    bin_mask = (mask >= thresh).astype(np.uint8) * 255

    # Use smaller kernels for smaller mask
    if morph_open > 0:
        k_open = max(2, int(morph_open * (mask_w / 416)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k)

    if morph_close > 0:
        k_close = max(3, int(morph_close * (mask_w / 416)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k)

    scale, _, _ = pad_info
    mask_to_unet_ratio = mask_w / input_size[0] # assume square UNet input for ratio
    scaled_min_area = int(min_area_px * (scale * mask_to_unet_ratio)**2)

    contours, _ = cv2.findContours(
        bin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    dets = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < scaled_min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        
        # Scale bbox from mask resolution (incorporating letterbox offsets) back to original resolution
        scale, x_offset, y_offset = pad_info

        # Extrapolate box coordinates from native mask size to target UNet input size
        # mask is e.g. 1/8th of target_h, target_w
        target_w, target_h = input_size
        unscale_w = target_w / mask_w
        unscale_h = target_h / mask_h
        
        # First get bounding box relative to the 320x320 UNet input
        unet_x1 = x * unscale_w
        unet_y1 = y * unscale_h
        unet_x2 = (x + w) * unscale_w
        unet_y2 = (y + h) * unscale_h

        # Now remove the padding offset and scale back to original image size
        x1 = int((unet_x1 - x_offset) / scale)
        y1 = int((unet_y1 - y_offset) / scale)
        x2 = int((unet_x2 - x_offset) / scale)
        y2 = int((unet_y2 - y_offset) / scale)

        # Clip to original image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        # Average score from the native-res mask
        roi = mask[y:y+h, x:x+w]
        score = float(np.mean(roi)) if roi.size > 0 else 0.0

        dets.append(
            {
                "bbox": (x1, y1, x2, y2),
                "score": score,
                "area": area, # relative area in mask
            }
        )

    dets.sort(key=lambda d: d["area"], reverse=True)
    dets = dets[:max_faces]

    for d in dets:
        d.pop("area", None)

    return dets


# -------------------------------------------------
# TFLITE UNET DETECTOR
# -------------------------------------------------

class TFLiteUNetDetector(Detector):

    def __init__(
        self,
        model_path: str,
        rgb: bool = True,
        mask_thresh: float = 0.5,
        conf_thresh: float = 0.3,
        min_area_px: int = 1500,
        max_faces: int = 5,
        num_threads: int = 4,
        use_gpu: bool = True,
    ):

        delegates, backend = make_delegate(use_gpu)

        self.interp = Interpreter(
            model_path=model_path,
            experimental_delegates=delegates,
            num_threads=num_threads,
        )
        print(f"[TFLITE DETECTOR] Backend: {backend}")
        self.interp.allocate_tensors()
        self._lock = threading.Lock()

        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

        self.rgb = rgb
        self.mask_thresh = mask_thresh
        self.conf_thresh = conf_thresh
        self.min_area_px = min_area_px
        self.max_faces = max_faces

        # 🔥 AUTO DETECT MODEL INPUT SIZE
        input_shape = self.inp["shape"]

        # Expect [1, H, W, C] or [1, C, H, W]
        if input_shape[1] == 3:
            # NCHW
            self.input_layout = "NCHW"
            self.input_size = (input_shape[2], input_shape[3])
        else:
            # NHWC
            self.input_layout = "NHWC"
            self.input_size = (input_shape[1], input_shape[2])

        print(f"[UNET] Input size detected: {self.input_size}")

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray):

        img = frame_bgr

        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox resize
        target_w, target_h = self.input_size
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas filled with zeros (padding)
        canvas = np.zeros((target_h, target_w, 3), dtype=img.dtype)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        img = canvas

        self._last_scale = scale
        self._last_x_offset = x_offset
        self._last_y_offset = y_offset

        # Save the 320x320 image given to UNet for debugging
        # debug_save_path = "unet_320x320_input.jpg"
        # save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if self.rgb else img
        # cv2.imwrite(debug_save_path, save_img)

        dtype = self.inp["dtype"]

        if dtype == np.float32:
            x = img.astype(np.float32) / 255.0

        elif dtype in (np.uint8, np.int8):
            scale, zero = self.inp.get("quantization", (1.0, 0))
            x = img.astype(np.float32)
            x = x / scale + zero

            if dtype == np.uint8:
                x = np.clip(x, 0, 255).astype(np.uint8)
            else:
                x = np.clip(x, -128, 127).astype(np.int8)

        else:
            x = img.astype(dtype)

        # Layout handling
        if self.input_layout == "NCHW":
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, axis=0)
        else:
            x = np.expand_dims(x, axis=0)

        return x

    # -------------------------------------------------
    # DETECT
    # -------------------------------------------------

    def detect(self, frame_bgr: np.ndarray):

        if frame_bgr is None:
            return []

        orig_h, orig_w = frame_bgr.shape[:2]
        # print("orig_h: ", orig_h)
        # print("orig_w: ", orig_w)

        x = self._preprocess(frame_bgr)
        #processed_h, processed_w = x.shape[0]
        # print("after preprocess: ")
        # print("h: ", x.shape[0])
        # print("w: ", x.shape[1])
        # print("w: ", x.shape[2])
        # print("w: ", x.shape[3])
        
        with self._lock:
            self.interp.set_tensor(self.inp["index"], x)
            self.interp.invoke()

            out = self.interp.get_tensor(self.out["index"]).copy()

        out = _dequantize(out, self.out)

        mask_small = _get_2d_mask(out)

        # Process bboxes directly at mask resolution, scaling bboxes back to (orig_w, orig_h)
        # Pass input size to mask_to_bboxes via globals or refactor
        # We need target_w/target_h in mask_to_bboxes. Let's pass it via pad_info.
        dets = mask_to_bboxes(
            mask_small,
            orig_size=(orig_w, orig_h),
            pad_info=(self._last_scale, self._last_x_offset, self._last_y_offset),
            input_size=self.input_size,
            thresh=self.mask_thresh,
            min_area_px=self.min_area_px,
            max_faces=self.max_faces,
        )

        dets = [d for d in dets if d["score"] >= self.conf_thresh]

        return self._validate_detections(frame_bgr, dets)
