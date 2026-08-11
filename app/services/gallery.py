import numpy as np
from sqlalchemy import text
import logging

from ..utils.cosine import pack_f32, unpack_f32

logger = logging.getLogger(__name__)


class Gallery:
    """
    Persistent embedding gallery backed by SQLite.
    """

    def __init__(self, db):
        self.db = db

    # ---------------------------------------------------------
    # LOAD ALL PROTOTYPES
    # ---------------------------------------------------------

    async def load_prototypes(self):

        async with self.db.begin() as conn:
            rows = (
                await conn.execute(
                    text("""
                        SELECT p.employee_id, p.embedding, e.name
                        FROM prototypes p
                        JOIN employees e ON p.employee_id = e.id
                        ORDER BY p.employee_id ASC
                    """)
                )
            ).fetchall()

        if not rows:
            return [], None, {}

        emp_ids = []
        protos = []
        emp_names_dict = {}
        detected_dim = None

        for emp_id, blob, name in rows:

            try:
                emb = unpack_f32(blob)
            except Exception as e:
                logger.warning(f"[GALLERY] Failed to unpack {emp_id}: {e}")
                continue

            if emb is None or emb.size == 0:
                logger.warning(f"[GALLERY] Empty embedding for {emp_id}")
                continue

            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            if not np.isfinite(emb).all():
                logger.warning(f"[GALLERY] NaN/Inf in embedding for {emp_id}")
                continue

            # -----------------------------------------
            # DETECT GLOBAL EMBEDDING DIMENSION
            # -----------------------------------------

            if detected_dim is None:
                detected_dim = emb.shape[0]

            if emb.shape[0] != detected_dim:
                logger.error(
                    f"[GALLERY] Dimension mismatch detected in DB: "
                    f"{emp_id} has {emb.shape[0]} but expected {detected_dim}"
                )
                raise RuntimeError(
                    "Inconsistent embedding dimensions found in DB."
                )

            # Defensive normalization
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                logger.warning(f"[GALLERY] Zero norm embedding for {emp_id}")
                continue

            emb = emb / norm

            emp_ids.append(emp_id)
            protos.append(emb)
            emp_names_dict[emp_id] = name

        if not protos:
            return [], None, {}

        proto_matrix = np.stack(protos, axis=0)

        logger.info(
            f"[GALLERY] Loaded {len(emp_ids)} prototypes "
            f"(dim={proto_matrix.shape[1]})"
        )

        return emp_ids, proto_matrix, emp_names_dict

    # ---------------------------------------------------------
    # INSERT OR UPDATE PROTOTYPE
    # ---------------------------------------------------------

    async def upsert_prototype(self, employee_id: str, emb: np.ndarray):

        if not isinstance(employee_id, str) or not employee_id:
            raise ValueError("Invalid employee_id")

        if emb is None:
            raise ValueError("Embedding is None")

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)

        if emb.ndim != 1:
            raise ValueError("Embedding must be 1D")

        if not np.isfinite(emb).all():
            raise ValueError("Embedding contains NaN/Inf")

        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            raise ValueError("Embedding norm too small")

        emb = emb / norm

        blob = pack_f32(emb)

        async with self.db.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO prototypes (employee_id, embedding)
                    VALUES (:id, :emb)
                    ON CONFLICT(employee_id)
                    DO UPDATE SET embedding = excluded.embedding
                """),
                {
                    "id": employee_id,
                    "emb": blob,
                },
            )

        logger.info(f"[GALLERY] Upserted prototype for {employee_id}")

    # ---------------------------------------------------------
    # DELETE PROTOTYPE
    # ---------------------------------------------------------

    async def delete_prototype(self, employee_id: str):

        async with self.db.begin() as conn:
            await conn.execute(
                text("""
                    DELETE FROM prototypes
                    WHERE employee_id = :id
                """),
                {"id": employee_id},
            )

        logger.info(f"[GALLERY] Deleted prototype for {employee_id}")
