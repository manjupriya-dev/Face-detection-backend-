from pathlib import Path
import asyncio
import io
import json
import uuid
import base64
import time
import yaml
import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import Optional
import os
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.db import make_engine, init_db
from app.services.gallery import Gallery
from app.services.enrollment import enroll_employee
from app.services.stream_manager import StreamManager
from app.pipeline.capture import OpenCVCapture
from app.pipeline.runtime import PipelineRuntime
from app.pipeline.recognizer import OpenSetMatcher
from app.pipeline.preprocessor import crop_face
from app.models.model_factory import build_models
from app.utils.cosine import pack_f32


# -------------------------------------------------
# APP INIT
# -------------------------------------------------

app = FastAPI()
state = {}

# -------------------------------------------------
# PREVIEW SESSION STORE
# In-memory dict: session_id → { created, employee_id, name, faces: [...] }
# Each face: { token, face_bgr, filename }
# TTL: 10 minutes
# -------------------------------------------------

_preview_sessions: dict[str, dict] = {}
_SESSION_TTL_SECONDS = 600


def _purge_expired_sessions():
    """Remove sessions older than TTL to free memory."""
    cutoff = time.time() - _SESSION_TTL_SECONDS
    expired = [sid for sid, s in _preview_sessions.items() if s["created"] < cutoff]
    for sid in expired:
        del _preview_sessions[sid]


# -------------------------------------------------
# CORS
# -------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# STARTUP
# -------------------------------------------------

@app.on_event("startup")
async def startup():

    BASE_DIR = Path(__file__).resolve().parent
    cfg = yaml.safe_load((BASE_DIR / "config.yaml").read_text())

    runtime_val = cfg.get("models", {}).get("runtime", "tflite")
    
    # Selection of database based on runtime
    if runtime_val == "onnx":
        cfg["db"]["path"] = "data/face_edge_open.sqlite3"
    else:
        # All other runtimes (custom, tflite, onnx+tflite) use the main database
        cfg["db"]["path"] = "data/face_edge.sqlite3"

    if "debug_capture" in cfg and "save_dir" in cfg["debug_capture"]:
        if "{runtime}" in cfg["debug_capture"]["save_dir"]:
            cfg["debug_capture"]["save_dir"] = cfg["debug_capture"]["save_dir"].format(runtime=runtime_val)

    if "enrollment" in cfg and "debug_dir" in cfg["enrollment"]:
        if cfg["enrollment"]["debug_dir"] is not None and "{runtime}" in cfg["enrollment"]["debug_dir"]:
            cfg["enrollment"]["debug_dir"] = cfg["enrollment"]["debug_dir"].format(runtime=runtime_val)

    if "event_log" in cfg and "csv_path" in cfg["event_log"]:
        if cfg["event_log"]["csv_path"] is not None and "{runtime}" in cfg["event_log"]["csv_path"]:
            cfg["event_log"]["csv_path"] = cfg["event_log"]["csv_path"].format(runtime=runtime_val)

    state["cfg"] = cfg

    # ---------------- HLS DIR ----------------

    hls_dir = Path(cfg["streaming"]["hls_output_dir"])
    hls_dir.mkdir(parents=True, exist_ok=True)

    app.mount(
        "/stream/hls",
        StaticFiles(directory=str(hls_dir)),
        name="hls"
    )

    # ---------------- DATABASE ----------------

    db = make_engine(cfg["db"]["path"])
    await init_db(db)
    state["db"] = db

    gallery = Gallery(db)
    state["gallery"] = gallery

    # ---------------- MATCHER ----------------

    matcher = OpenSetMatcher(**cfg["recognition"])
    state["matcher"] = matcher

    # ---------------- MODELS ----------------

    detector, embedder, landmarker = build_models(cfg)

    state["detector"] = detector
    state["embedder"] = embedder
    state["landmarker"] = landmarker

    print(f"[MODEL] Runtime: {cfg['models']['runtime']}")
    print(f"[MODEL] Embed dimension: {embedder.embed_dim}")

    # ---------------- LOAD GALLERY ----------------

    emp_ids, protos, emp_names = await gallery.load_prototypes()

    if protos is not None:
        if protos.shape[1] != embedder.embed_dim:
            raise RuntimeError(
                "Embedding dimension mismatch.\n"
                "Delete SQLite DB when switching models."
            )

        matcher.update_gallery(emp_ids, protos, emp_names)

    # ---------------- CAMERA ----------------

    cam_cfg = cfg["camera"]

    capture = OpenCVCapture(
        cam_cfg["source"],
        cam_cfg["width"],
        cam_cfg["height"],
        cam_cfg["fps"],
    )

    state["capture"] = capture

    # ---------------- STREAM ----------------

    stream_manager = StreamManager(cfg)
    stream_manager.start_pipe_encoder()
    state["stream_manager"] = stream_manager

    # ---------------- PIPELINE ----------------

    loop = asyncio.get_running_loop()

    runtime = PipelineRuntime(
        capture=capture,
        detector=detector,
        embedder=embedder,
        landmarker=landmarker,
        matcher=matcher,
        cfg=cfg,
        db=db,
        stream_manager=stream_manager,
        loop=loop,
    )

    runtime.start()
    state["runtime"] = runtime

    print("[SYSTEM] AI pipeline started")


# -------------------------------------------------
# SHUTDOWN
# -------------------------------------------------

@app.on_event("shutdown")
async def shutdown():

    if state.get("runtime"):
        state["runtime"].stop()

    if state.get("stream_manager"):
        state["stream_manager"].cleanup()

    if state.get("db"):
        await state["db"].dispose()


# -------------------------------------------------
# STATUS
# -------------------------------------------------

@app.get("/status")
async def status():
    rt = state.get("runtime")

    strike_rate = 0.0
    try:
        if state.get("db"):
            async with state["db"].begin() as conn:
                row = (await conn.execute(
                    text("SELECT SUM(faces_identified), SUM(embedding_generated) FROM performance_logs")
                )).fetchone()
                if row and row[1]:
                    strike_rate = (row[0] / row[1]) * 100.0
    except Exception:
        pass

    return {
        "camera_active": rt is not None,
        "stats": rt.stats if rt else None,
        "gallery_size": len(state["matcher"].emp_ids),
        "runtime": state["cfg"]["models"]["runtime"],
        "strike_rate_percent": float(f"{strike_rate:.2f}"),
    }


# -------------------------------------------------
# LOGS
# -------------------------------------------------

@app.get("/logs")
async def get_logs(limit: int = 100):

    async with state["db"].begin() as conn:
        result = await conn.execute(
            text("""
                SELECT
                    e.id,
                    e.ts,
                    e.camera_id,
                    e.track_id,
                    e.result,
                    e.score,
                    emp.name
                FROM events e
                LEFT JOIN employees emp
                    ON e.result = emp.id
                ORDER BY e.id DESC
                LIMIT :limit
            """),
            {"limit": limit}
        )
        rows = result.fetchall()

    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "camera_id": r[2],
            "track_id": r[3],
            "employee_id": r[4],
            "employee_name": r[6] if r[6] else "Unknown",
            "score": r[5],
        }
        for r in rows
    ]


# -------------------------------------------------
# ENROLL  (original — unchanged)
# -------------------------------------------------

@app.post("/enroll")
async def enroll(
    employee_id: str = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No images provided")

        images = []

        for f in files:
            data = await f.read()

            try:
                pil_img = Image.open(io.BytesIO(data))
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is not None:
                images.append(img)

        if not images:
            raise HTTPException(status_code=400, detail="Invalid images")

        enroll_cfg = state["cfg"].get("enrollment", {})

        proto = await enroll_employee(
            state["db"],
            state["gallery"],
            state["detector"],
            state["landmarker"],
            state["embedder"],
            employee_id,
            name,
            images,
            min_detection_score=enroll_cfg.get("min_detection_score", 0.45),
            min_face_size=enroll_cfg.get("min_face_size", 60),
            debug_dir=enroll_cfg.get("debug_dir"),
        )

        emp_ids, protos, emp_names = await state["gallery"].load_prototypes()

        if protos is not None:
            if protos.shape[1] != state["embedder"].embed_dim:
                raise RuntimeError("Embedding dimension mismatch after enrollment.")
            state["matcher"].update_gallery(emp_ids, protos, emp_names)

        return {
            "employee_id": employee_id,
            "name": name,
            "prototype_norm": float(np.linalg.norm(proto)),
            "num_images_processed": len(images),
        }

    except HTTPException:
        raise
    except Exception as e:
        print("ENROLL ERROR:", str(e))
        raise HTTPException(status_code=422, detail=str(e))


# -------------------------------------------------
# ENROLL — STEP 1: PREVIEW
# Upload images → detect & crop faces only.
# Embedding model is NOT called here.
# Returns base64 cropped face previews + per-image tokens.
# Pass existing session_id to append more images (Add More flow).
# -------------------------------------------------

@app.post("/enroll/preview")
async def enroll_preview(
    employee_id: str = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    STEP 1 of 2 — Detect & crop faces only. Embedding model NOT called.
    Returns base64 JPEG previews + a unique token per image.
    """
    _purge_expired_sessions()

    if not files:
        raise HTTPException(status_code=400, detail="No images provided")

    # Reuse existing session (Add More flow) or create a fresh one
    if session_id and session_id in _preview_sessions:
        session = _preview_sessions[session_id]
        session["employee_id"] = employee_id
        session["name"] = name
    else:
        session_id = str(uuid.uuid4())
        session = {
            "created":     time.time(),
            "employee_id": employee_id,
            "name":        name,
            "faces":       [],  # list of { token, face_bgr, filename }
        }
        _preview_sessions[session_id] = session

    enroll_cfg = state["cfg"].get("enrollment", {})
    min_score  = enroll_cfg.get("min_detection_score", 0.45)
    min_size   = enroll_cfg.get("min_face_size", 60)

    previews = []

    for f in files:
        data     = await f.read()
        filename = f.filename or "unknown.jpg"

        # Decode with EXIF rotation correction
        img_bgr = None
        try:
            pil_img = Image.open(io.BytesIO(data))
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            arr     = np.frombuffer(data, np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            previews.append({
                "token":    None,
                "filename": filename,
                "status":   "error",
                "error":    "Cannot decode image",
                "face_b64": None,
            })
            continue

        # ── Detect & crop ONLY — embedder is NOT called ───────────────
        try:
            detections = state["detector"].detect(img_bgr)

            if not detections:
                raise ValueError("No face detected")

            # Pick the largest face if multiple detected
            det = sorted(
                detections,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
                reverse=True,
            )[0]

            bbox  = det["bbox"]
            score = det.get("score", 1.0)

            if score < min_score:
                raise ValueError(f"Low confidence ({score:.2f} < {min_score})")

            x1, y1, x2, y2 = bbox
            w_face = x2 - x1
            h_face = y2 - y1

            if w_face < min_size or h_face < min_size:
                raise ValueError(
                    f"Face too small ({int(w_face)}x{int(h_face)} px, need >= {min_size} px)"
                )

            face_bgr = crop_face(img_bgr, bbox)
            if face_bgr is None:
                raise ValueError("Face crop failed")

            # Encode cropped face to JPEG for preview transport
            _, buf   = cv2.imencode(".jpg", face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            face_b64 = base64.b64encode(buf.tobytes()).decode()
            token    = str(uuid.uuid4())

            # Store the BGR crop in session for later embedding
            session["faces"].append({
                "token":    token,
                "face_bgr": face_bgr,
                "filename": filename,
            })

            previews.append({
                "token":    token,
                "filename": filename,
                "status":   "ok",
                "error":    None,
                "face_b64": face_b64,
            })

        except Exception as ex:
            previews.append({
                "token":    None,
                "filename": filename,
                "status":   "error",
                "error":    str(ex),
                "face_b64": None,
            })

    return {
        "session_id":             session_id,
        "employee_id":            employee_id,
        "name":                   name,
        "previews":               previews,
        "total_faces_in_session": len(session["faces"]),
    }


# -------------------------------------------------
# ENROLL — STEP 2: CONFIRM
# Receives approved token list -> runs embedder -> stores prototype.
# This is the ONLY place the embedding model is called.
# -------------------------------------------------

@app.post("/enroll/confirm")
async def enroll_confirm(
    session_id:      str = Form(...),
    approved_tokens: str = Form(...),   # JSON array of token strings
):
    """
    STEP 2 of 2 — Run the embedding model on user-approved face crops
    and store the prototype in the database.
    Removed / rejected images are simply ignored.
    """
    _purge_expired_sessions()

    session = _preview_sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Preview session expired or not found. Please re-upload your images.",
        )

    try:
        token_list: list[str] = json.loads(approved_tokens)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="approved_tokens must be a JSON array of strings",
        )

    if not token_list:
        raise HTTPException(status_code=400, detail="No approved images — nothing to enroll")

    token_set      = set(token_list)
    approved_faces = [f for f in session["faces"] if f["token"] in token_set]

    if not approved_faces:
        raise HTTPException(
            status_code=400,
            detail="None of the approved tokens matched session data. Session may have expired.",
        )

    employee_id = session["employee_id"]
    name        = session["name"]
    enroll_cfg  = state["cfg"].get("enrollment", {})
    debug_dir   = enroll_cfg.get("debug_dir")

    try:
        # ── Run the embedding model on approved faces only ─────────────
        embeddings = []

        for face_entry in approved_faces:
            face_bgr = face_entry["face_bgr"]
            try:
                # [DEBUG] Save raw face crop (interpreted as post-unet) during enrollment
                if debug_dir and employee_id and name:
                    try:
                        base_dir = os.path.join(debug_dir, f"{employee_id}_{name}")
                        unet_dir = os.path.join(base_dir, "After_unet")
                        os.makedirs(unet_dir, exist_ok=True)
                        ts_str = f"{int(time.time() * 1000)}"
                        filename = f"{employee_id}_{name}_{ts_str}.jpg"
                        cv2.imwrite(os.path.join(unet_dir, filename), face_bgr)
                    except Exception as e:
                        print(f"Failed to save debug face_raw: {e}")

                # Align with landmarker or plain resize as fallback
                if state["landmarker"] is not None:
                    aligned = state["landmarker"].align(face_bgr)
                    if aligned is None:
                        raise ValueError("Landmark alignment failed")
                else:
                    aligned = cv2.resize(face_bgr, (112, 112))

                # [DEBUG] Save aligned face crop during enrollment
                if debug_dir and employee_id and name:
                    try:
                        align_dir = os.path.join(base_dir, "Align")
                        os.makedirs(align_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(align_dir, filename), aligned)
                    except Exception as e:
                        print(f"Failed to save debug face_aligned: {e}")

                emb = state["embedder"].embed(aligned)

                if emb is None or emb.size == 0:
                    raise ValueError("Empty embedding")
                if not np.isfinite(emb).all():
                    raise ValueError("NaN/Inf in embedding")

                emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                embeddings.append(emb)

            except Exception as ex:
                print(f"[CONFIRM] Skipping face '{face_entry['filename']}': {ex}")

        if not embeddings:
            raise HTTPException(
                status_code=422,
                detail="No valid embeddings could be extracted from the approved images.",
            )

        # ── Build prototype ────────────────────────────────────────────
        proto       = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        norm        = np.linalg.norm(proto)
        photo_count = len(embeddings)

        if norm < 1e-9:
            raise HTTPException(status_code=422, detail="Prototype has zero norm")

        proto /= norm

        if proto.shape[0] != state["embedder"].embed_dim:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Prototype dimension mismatch: "
                    f"expected {state['embedder'].embed_dim}, got {proto.shape[0]}"
                ),
            )

        # ── Persist to database ────────────────────────────────────────
        blob = pack_f32(proto)

        # Use system local time
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        async with state["db"].begin() as conn:
            existing_emp = (await conn.execute(
                text("SELECT id FROM employees WHERE id = :id"),
                {"id": employee_id},
            )).fetchone()

            if existing_emp:
                # Update name and photo_count for re-enrollment
                await conn.execute(
                    text("""
                        UPDATE employees
                        SET name = :name, photo_count = :photo_count
                        WHERE id = :id
                    """),
                    {"id": employee_id, "name": name, "photo_count": photo_count},
                )
            else:
                # New employee — include all NOT NULL columns
                await conn.execute(
                    text("""
                        INSERT INTO employees (id, name, created_at, photo_count)
                        VALUES (:id, :name, :now, :photo_count)
                    """),
                    {"id": employee_id, "name": name, "photo_count": photo_count, "now": now},
                )

            # Upsert prototype — handles both new and re-enrollment
            await conn.execute(
                text("""
                    INSERT INTO prototypes (employee_id, embedding, updated_at)
                    VALUES (:id, :emb, :now)
                    ON CONFLICT(employee_id)
                    DO UPDATE SET
                        embedding  = excluded.embedding,
                        updated_at = :now
                """),
                {"id": employee_id, "emb": blob, "now": now},
            )

        # ── Reload gallery into matcher ────────────────────────────────
        emp_ids, protos_mat, emp_names = await state["gallery"].load_prototypes()
        if protos_mat is not None:
            if protos_mat.shape[1] != state["embedder"].embed_dim:
                raise RuntimeError("Embedding dimension mismatch after enrollment.")
            state["matcher"].update_gallery(emp_ids, protos_mat, emp_names)
        else:
            state["matcher"].update_gallery([], None, {})

        # ── Clean up session from memory ───────────────────────────────
        _preview_sessions.pop(session_id, None)

        return {
            "employee_id":          employee_id,
            "name":                 name,
            "num_images_processed": photo_count,
            "prototype_norm":       float(np.linalg.norm(proto)),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[CONFIRM ERROR] {e}")
        raise HTTPException(status_code=422, detail=str(e))


# -------------------------------------------------
# ENROLL — CANCEL PREVIEW SESSION
# -------------------------------------------------

@app.delete("/enroll/preview/{session_id}")
async def cancel_preview(session_id: str):
    """Discard a preview session and free its memory (Cancel button)."""
    _preview_sessions.pop(session_id, None)
    return {"status": "cancelled", "session_id": session_id}


# -------------------------------------------------
# EMPLOYEES LIST
# -------------------------------------------------

@app.get("/employees")
async def get_employees():
    async with state["db"].begin() as conn:
        result = await conn.execute(
            text("""
                SELECT
                    e.id            AS employee_id,
                    e.name,
                    e.active,
                    e.created_at,
                    p.updated_at    AS enrollment_date,
                    e.photo_count
                FROM employees e
                LEFT JOIN prototypes p ON p.employee_id = e.id
                WHERE e.active = 1
                ORDER BY e.created_at DESC
            """)
        )
        rows = result.fetchall()

    return [
        {
            "employee_id":     r[0],
            "name":            r[1],
            "active":          bool(r[2]),
            "enrollment_date": (r[4] or r[3] or "")[:10],
            "enrollment_time": (r[4] or r[3] or "")[11:19],
            "photo_count":     r[5] if r[5] is not None else 0,
        }
        for r in rows
    ]


# -------------------------------------------------
# SINGLE EMPLOYEE  (lookup + delete)
# -------------------------------------------------

@app.get("/employee/{employee_id}")
async def get_employee(employee_id: str):
    async with state["db"].begin() as conn:
        result = await conn.execute(
            text("SELECT id, name, active, created_at FROM employees WHERE id = :id"),
            {"id": employee_id}
        )
        row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Employee not found")

    return {
        "employee_id": row[0],
        "name":        row[1],
        "active":      bool(row[2]),
        "created_at":  row[3],
    }


@app.put("/employee/{employee_id}")
async def update_employee(employee_id: str, name: str = Form(...)):
    async with state["db"].begin() as conn:
        result = await conn.execute(
            text("SELECT id FROM employees WHERE id = :id"),
            {"id": employee_id}
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Employee not found")

        await conn.execute(
            text("UPDATE employees SET name = :name WHERE id = :id"),
            {"id": employee_id, "name": name.strip()}
        )

    return {"employee_id": employee_id, "name": name.strip()}


@app.delete("/employee/{employee_id}")
async def delete_employee(employee_id: str):
    async with state["db"].begin() as conn:
        result = await conn.execute(
            text("SELECT id FROM employees WHERE id = :id"),
            {"id": employee_id}
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Employee not found")

        await conn.execute(
            text("DELETE FROM employees WHERE id = :id"),
            {"id": employee_id}
        )

    emp_ids, protos, emp_names = await state["gallery"].load_prototypes()
    if protos is not None:
        state["matcher"].update_gallery(emp_ids, protos, emp_names)
    else:
        state["matcher"].update_gallery([], None, {})

    return {"deleted": employee_id}