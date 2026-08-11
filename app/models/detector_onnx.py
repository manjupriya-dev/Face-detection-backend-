import cv2
import numpy as np

from .detector_base import Detector


class ONNXYuNetDetector(Detector):
    """
    YuNet ONNX detector using OpenCV FaceDetectorYN.
    """

    def __init__(
        self,
        model_path: str,
        score_thresh: float = 0.6,
        nms_thresh: float = 0.3,
        top_k: int = 5000,
    ):
        self.model_path = model_path
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k

        # Initialize with dummy size (will update per frame)
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_thresh,
            nms_thresh,
            top_k,
        )

        self._last_input_size = None

    # -------------------------------------------------
    # DETECT
    # -------------------------------------------------

    def detect(self, frame_bgr: np.ndarray):

        if frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]

        # Update input size only if frame size changed
        if self._last_input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._last_input_size = (w, h)

        try:
            _, faces = self.detector.detect(frame_bgr)
        except Exception:
            # OpenCV sometimes throws if frame corrupted
            return []

        if faces is None or len(faces) == 0:
            return []

        detections = []

        for f in faces:
            # YuNet output format:
            # [x, y, w, h, score, ...landmarks]
            x, y, bw, bh, score = f[:5]

            if score < self.score_thresh:
                continue

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + bw)
            y2 = int(y + bh)

            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "score": float(score),
                }
            )

        # Enforce base class validation
        return self._validate_detections(frame_bgr, detections)
