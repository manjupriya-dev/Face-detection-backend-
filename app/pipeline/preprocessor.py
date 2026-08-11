import numpy as np
import cv2

# ------------------------------------------------------------
# 1. QUALITY CHECKS (Gatekeeping)
# ------------------------------------------------------------

def is_sharp_enough(face_crop: np.ndarray, threshold: float = 100.0) -> bool:
    """Uses Laplacian variance to detect motion blur."""
    if face_crop is None or face_crop.size == 0:
        return False
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= threshold

def is_lighting_ok(face_crop: np.ndarray, min_br: int = 40, max_br: int = 225) -> bool:
    """Ensures face isn't too dark or overexposed by IR LEDs."""
    if face_crop is None or face_crop.size == 0:
        return False
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return min_br < avg_brightness < max_br

def crop_face(frame_bgr: np.ndarray, bbox_xyxy, pad: float = 0.25):
    """
    Crops with extra padding. Alignment works better if it has 
    some context around the face landmarks.
    """
    if frame_bgr is None or bbox_xyxy is None:
        return None
 
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    bw, bh = x2 - x1, y2 - y1
 
    if bw <= 0 or bh <= 0:
        return None
 
    px, py = int(bw * pad), int(bh * pad)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
 
    return frame_bgr[cy1:cy2, cx1:cx2]

# ------------------------------------------------------------
# 2. SMART RESIZING (Aspect-Ratio Safe)
# ------------------------------------------------------------

def resize_with_aspect_ratio(face_crop, target_size=112):
    """
    Stretches or shrinks the crop to fit target_size.
    Uses black padding (letterboxing) to maintain proportions.
    """
    if face_crop is None or face_crop.size == 0:
        return None
 
    h, w = face_crop.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
 
    # Use CUBIC for stretching small faces, AREA for shrinking large ones
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized_img = cv2.resize(face_crop, (new_w, new_h), interpolation=interp)
 
    # Create black square canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
 
    # Center the face on the canvas
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
 
    return canvas

# ------------------------------------------------------------
# 3. MASTER PROCESSING FUNCTION
# ------------------------------------------------------------

def get_final_face_patch(frame_bgr, bbox_xyxy, landmarker=None, cfg_recognition=None):
    """
    The main entry point called by runtime.py.
    """
    # A. Initial Crop - using our helper above
    raw_crop = crop_face(frame_bgr, bbox_xyxy, pad=0.1)
    if raw_crop is None:
        return None, None, "Crop Failed"

    # Size Check (from config)
    min_size = cfg_recognition.get("min_face_px", 80) if cfg_recognition else 80
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None, raw_crop, "Too Small"

    # B. Blur Check (Discard bad frames)
    if not is_sharp_enough(raw_crop, threshold=100.0):
        return None, raw_crop, "Blurry"
 
    # C. Lighting Check (Essential for CP-PLUS IR)
    if not is_lighting_ok(raw_crop):
        return None, raw_crop, "Bad Lighting"
 
    # D. Scaling & Alignment
    # 1st Priority: High-quality alignment if landmarker is active
    if landmarker is not None:
        aligned_face = landmarker.align(raw_crop)
        if aligned_face is not None:
            return aligned_face, raw_crop, "Aligned"

    # 2nd Priority: Fallback to Letterbox Resize (Aspect-ratio safe)
    final_patch = resize_with_aspect_ratio(raw_crop, target_size=112)
    return final_patch, raw_crop, "Resized"
