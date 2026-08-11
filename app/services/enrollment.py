import numpy as np
import logging
import time
import os
from datetime import datetime
from sqlalchemy import text
import cv2

from ..pipeline.preprocessor import crop_face
from ..utils.cosine import pack_f32

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# STRICT FACE PIPELINE (Detector → Landmarker → Embedder)
# ------------------------------------------------------------

def process_single_image(
    img,
    detector,
    landmarker,
    embedder,
    min_detection_score=0.45,
    min_face_size=60,
    debug_dir=None,
    employee_id=None,
    employee_name=None,
):
    """Run the strict face pipeline on one image and return a 512-D embedding.

    ``min_detection_score`` and ``min_face_size`` are normally supplied by the
    caller from ``config.yaml → enrollment`` so they can be tuned centrally.
    The defaults here are conservative fallbacks in case the caller omits them.
    """

    detections = detector.detect(img)

    if not detections:
        raise RuntimeError("No face detected")

    # If multiple faces detected, pick largest one
    if len(detections) > 1:
        detections = sorted(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True
        )

    det   = detections[0]
    bbox  = det["bbox"]
    score = det.get("score", 1.0)

    if score < min_detection_score:
        raise RuntimeError(f"Low detection confidence: {score:.2f} (need ≥ {min_detection_score})")

    x1, y1, x2, y2 = bbox

    if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
        raise RuntimeError(
            f"Face too small: {int(x2-x1)}×{int(y2-y1)} px (need ≥ {min_face_size} px)"
        )

    face = crop_face(img, bbox)
    if face is None:
        raise RuntimeError("Face crop failed")

    # [DEBUG] Save raw face crop (interpreted as post-unet) during enrollment
    if debug_dir and employee_id and employee_name:
        try:
            base_dir = os.path.join(debug_dir, f"{employee_id}_{employee_name}")
            unet_dir = os.path.join(base_dir, "After_unet")
            os.makedirs(unet_dir, exist_ok=True)
            ts_str = f"{int(time.time() * 1000)}"
            filename = f"{employee_id}_{employee_name}_{ts_str}.jpg"
            cv2.imwrite(os.path.join(unet_dir, filename), face)
        except Exception as e:
            logger.warning(f"Failed to save debug face_raw: {e}")

    if landmarker is not None:
        aligned = landmarker.align(face)
        if aligned is None:
            raise RuntimeError("Landmark alignment failed")
    else:
        aligned = cv2.resize(face, (112, 112))

    # [DEBUG] Save aligned face crop during enrollment
    if debug_dir and employee_id and employee_name:
        try:
            align_dir = os.path.join(base_dir, "Align")
            os.makedirs(align_dir, exist_ok=True)
            cv2.imwrite(os.path.join(align_dir, filename), aligned)
        except Exception as e:
            logger.warning(f"Failed to save debug face_aligned: {e}")

    emb = embedder.embed(aligned)

    if emb is None or emb.size == 0:
        raise RuntimeError("Empty embedding")

    if not np.isfinite(emb).all():
        raise RuntimeError("Embedding contains NaN/Inf")

    emb = np.asarray(emb, dtype=np.float32).reshape(-1)

    if emb.shape[0] != embedder.embed_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch. Expected {embedder.embed_dim}, got {emb.shape[0]}"
        )

    return emb

# ------------------------------------------------------------
# ENROLL EMPLOYEE
# ------------------------------------------------------------

async def enroll_employee(
    db,
    gallery,
    detector,
    landmarker,
    embedder,
    employee_id: str,
    name: str,
    images: list,
    min_detection_score: float = 0.45,
    min_face_size: int = 60,
    debug_dir=None,
):

    if not images:
        raise RuntimeError("No images provided")

    embeddings = []

    for img in images:
        try:
            emb = process_single_image(
                img,
                detector,
                landmarker,
                embedder,
                min_detection_score=min_detection_score,
                min_face_size=min_face_size,
                debug_dir=debug_dir,
                employee_id=employee_id,
                employee_name=name,
            )
            embeddings.append(emb)

        except Exception as e:
            logger.warning(f"Image rejected: {e}")
            print(f"[ENROLL] Image rejected: {e}")

    if len(embeddings) == 0:
        raise RuntimeError("No valid embeddings extracted")

    photo_count = len(embeddings)   # how many photos actually contributed

    # --------------------------------------------------------
    # CREATE 512D PROTOTYPE
    # --------------------------------------------------------

    proto = np.mean(np.stack(embeddings, axis=0), axis=0)
    proto = proto.astype(np.float32)

    norm = np.linalg.norm(proto)
    if norm < 1e-9:
        raise RuntimeError("Prototype has zero norm")

    proto = proto / norm

    if proto.shape[0] != embedder.embed_dim:
        raise RuntimeError("Final prototype dimension mismatch")

    logger.info(
        f"Prototype created for {employee_id} "
        f"using {len(embeddings)} images "
        f"(dim={proto.shape[0]})"
    )

    # --------------------------------------------------------
    # STORE IN DATABASE (STRICT 512D FLOAT32)
    # --------------------------------------------------------

    blob = pack_f32(proto)

    # Use system local time — reads your machine's clock directly
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async with db.begin() as conn:

        # Upsert employee (write photo_count from this enrollment batch)
        result = await conn.execute(
            text("SELECT id FROM employees WHERE id = :id"),
            {"id": employee_id},
        )
        existing = result.fetchone()

        if existing:
            await conn.execute(
                text("""
                    UPDATE employees
                    SET name = :name, photo_count = :photo_count
                    WHERE id = :id
                """),
                {"id": employee_id, "name": name, "photo_count": photo_count},
            )
        else:
            await conn.execute(
                text("""
                    INSERT INTO employees (id, name, created_at, photo_count)
                    VALUES (:id, :name, :now, :photo_count)
                """),
                {"id": employee_id, "name": name, "now": now, "photo_count": photo_count},
            )

        # Upsert prototype
        await conn.execute(
            text("""
                INSERT INTO prototypes (employee_id, embedding, updated_at)
                VALUES (:id, :emb, :now)
                ON CONFLICT(employee_id)
                DO UPDATE SET
                    embedding = excluded.embedding,
                    updated_at = :now
            """),
            {
                "id": employee_id,
                "emb": blob,
                "now": now,
            },
        )

    logger.info(
        f"Enrollment successful for {employee_id} "
        f"({photo_count} photos → 1 prototype)"
    )

    return proto