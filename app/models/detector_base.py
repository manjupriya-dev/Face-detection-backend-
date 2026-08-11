from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np


class Detector(ABC):
    """
    Abstract face detector interface.

    All detectors MUST return detections in this format:

    [
        {
            "bbox": (x1, y1, x2, y2),
            "score": float
        },
        ...
    ]
    """

    # -------------------------------------------------
    # REQUIRED METHOD
    # -------------------------------------------------

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        raise NotImplementedError

    # -------------------------------------------------
    # SAFE OUTPUT VALIDATION
    # -------------------------------------------------

    def _validate_detections(
        self,
        frame_bgr: np.ndarray,
        detections: List[Dict],
        min_area: int = 10,
        max_detections: int = 50,
    ) -> List[Dict]:

        if frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]

        if not isinstance(detections, list):
            raise RuntimeError("Detector must return a list")

        validated = []

        for det in detections:

            if not isinstance(det, dict):
                continue

            if "bbox" not in det or "score" not in det:
                continue

            bbox = det["bbox"]
            score = det["score"]

            if bbox is None or len(bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = map(int, bbox)
                score = float(score)
            except Exception:
                continue

            # Reject NaN / Inf scores
            if not np.isfinite(score):
                continue

            # Clip bbox
            x1, y1, x2, y2 = self._clip_bbox(
                x1, y1, x2, y2, w, h
            )

            # Validate geometry
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)

            if area < min_area:
                continue

            # Clamp score
            score = max(0.0, min(score, 1.0))

            validated.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "score": score,
                }
            )

        # Sort by confidence (high → low)
        validated.sort(key=lambda d: d["score"], reverse=True)

        # Limit number of detections
        if len(validated) > max_detections:
            validated = validated[:max_detections]

        return validated

    # -------------------------------------------------
    # BBOX CLIPPING
    # -------------------------------------------------

    @staticmethod
    def _clip_bbox(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        w: int,
        h: int,
    ) -> Tuple[int, int, int, int]:

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        return x1, y1, x2, y2
