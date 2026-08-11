import cv2
import os
import time
import threading
import queue
import logging
import collections
import csv
import json
import numpy as np
from pathlib import Path
from sqlalchemy import text
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING DEBUG FRAME BUFFER  (known / unknown routing)
# ─────────────────────────────────────────────────────────────────────────────

class DebugFrameBuffer:
    """Thread-safe, FIFO-capped buffer that saves annotated frames and ArcFace
    face patches sorted into ``known/`` and ``unknown/`` sub-directories.

    Directory layout::

        save_dir/
            known/
                frames/   ← full annotated frame (green bbox)
                faces/    ← 112×112 aligned patch
            unknown/
                frames/   ← full annotated frame (red bbox)
                faces/    ← 112×112 aligned patch
    """

    def __init__(self, save_dir: str, max_frames: int = 200):
        self._root      = Path(save_dir)
        self.max_frames = max_frames
        self._lock      = threading.Lock()
        self._counters  = {"known": 0, "unknown": 0}
        self._entries   = {
            "known":   collections.deque(),
            "unknown": collections.deque(),
        }
        for bucket in ("known", "unknown"):
            (self._root / bucket).mkdir(parents=True, exist_ok=True)

        logger.info(
            "[DebugBuffer] known → %s/known | unknown → %s/unknown | max %d each",
            self._root, self._root, self.max_frames,
        )

    # ------------------------------------------------------------------
    def push(self, frame: np.ndarray, frame_raw: np.ndarray, face_raw: "np.ndarray | None", face_patch: "np.ndarray | None", label: str, name: str = "Unknown"):
        """Save *frame* and optional *face_patch* into the bucket for *label*.

        ``label`` is routed to ``"known"`` for any recognised person, or
        ``"unknown"`` for unrecognised faces.  Returns individual milliseconds for performance logging.
        """
        bucket   = "unknown" if label == "unknown" else "known"
        image_full_ms = 0.0
        image_raw_ms = 0.0
        face_aligned_ms = 0.0
        face_raw_ms = 0.0

        with self._lock:
            self._counters[bucket] += 1
            idx = self._counters[bucket]

            if bucket == "known":
                bucket_dir = self._root / "known" / f"{label}_{name}"
                stem = f"{idx:06d}_{label}_{name}"
            else:
                bucket_dir = self._root / "unknown"
                stem = f"{idx:06d}"

            bucket_dir.mkdir(parents=True, exist_ok=True)

            frame_path = bucket_dir / f"{stem}_frame.jpg"
            frame_raw_path = bucket_dir / f"{stem}_frame_raw.jpg"
            face_path  = (bucket_dir / f"{stem}_face.jpg") \
                         if face_patch is not None else None
            face_raw_path  = (bucket_dir / f"{stem}_face_raw.jpg") \
                             if face_raw is not None else None

            entries = self._entries[bucket]
            if self.max_frames > 0 and len(entries) >= self.max_frames:
                old_frame, old_frame_raw, old_face, old_face_raw = entries.popleft()
                for p in (old_frame, old_frame_raw, old_face, old_face_raw):
                    if p is not None:
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass

            try:
                t0 = time.time()
                cv2.imwrite(str(frame_path), frame)
                image_full_ms = (time.time() - t0) * 1000

                t0 = time.time()
                cv2.imwrite(str(frame_raw_path), frame_raw)
                image_raw_ms = (time.time() - t0) * 1000

                if face_path is not None:
                    t0 = time.time()
                    cv2.imwrite(str(face_path), face_patch)
                    face_aligned_ms = (time.time() - t0) * 1000

                    if face_raw_path is not None:
                        t0 = time.time()
                        cv2.imwrite(str(face_raw_path), face_raw)
                        face_raw_ms = (time.time() - t0) * 1000

                entries.append((frame_path, frame_raw_path, face_path, face_raw_path))
            except Exception as exc:
                logger.warning("[DebugBuffer] Write failed: %s", exc)

        return image_full_ms, image_raw_ms, face_aligned_ms, face_raw_ms

    @property
    def count(self):
        with self._lock:
            return sum(len(d) for d in self._entries.values())


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING DEBUG VIDEO WRITER
# Records the annotated inference stream as segmented rolling MP4 files.
# ─────────────────────────────────────────────────────────────────────────────

class DebugVideoWriter:
    """Thread-safe rolling MP4 recorder for the annotated inference stream.

    Creates a new file every ``segment_sec`` seconds and deletes the oldest
    when the ``max_videos`` limit is exceeded.  Files are named by timestamp.
    """

    def __init__(
        self,
        video_dir:   str,
        fps:         float,
        width:       int,
        height:      int,
        segment_sec: float = 60.0,
        max_videos:  int   = 10,
    ):
        self._dir         = Path(video_dir)
        self._fps         = fps
        self._width       = width
        self._height      = height
        self._segment_sec = segment_sec
        self._max_videos  = max_videos
        self._lock        = threading.Lock()
        self._writer      = None
        self._seg_start   = 0.0
        self._files: collections.deque = collections.deque()

        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[VideoWriter] → %s | %.0fs segments | max %d files",
            self._dir, self._segment_sec, self._max_videos,
        )

    def write(self, frame: np.ndarray):
        """Write one frame, rolling to a new file when the segment ends."""
        with self._lock:
            now = time.time()
            if self._writer is None or (now - self._seg_start) >= self._segment_sec:
                self._roll(now)
            if self._writer is not None:
                try:
                    self._writer.write(frame)
                except Exception as exc:
                    logger.warning("[VideoWriter] Frame write failed: %s", exc)

    def close(self):
        """Flush and release the current video file."""
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception:
                    pass
                self._writer = None

    def _roll(self, now: float):
        """Close current segment and open a new one (caller holds lock)."""
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None

        ts     = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
        path   = self._dir / f"{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self._fps, (self._width, self._height))

        if not writer.isOpened():
            logger.error("[VideoWriter] Could not open %s", path)
            writer.release()
            return

        self._writer    = writer
        self._seg_start = now
        self._files.append(path)

        while len(self._files) > self._max_videos:
            old = self._files.popleft()
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass

        logger.info("[VideoWriter] New segment: %s", path)



from .preprocessor import crop_face, get_final_face_patch


class PipelineRuntime:

    def __init__(
        self,
        capture,
        detector,
        embedder,
        landmarker,   # may be None
        matcher,
        cfg,
        db,
        stream_manager,
        loop,
    ):

        self.capture = capture
        self.detector = detector
        self.embedder = embedder
        self.landmarker = landmarker
        self.matcher = matcher
        self.cfg = cfg
        self.db = db
        self.stream_manager = stream_manager
        self.loop = loop

        self.frame_q = queue.Queue(maxsize=2)         # capture → infer loop
        self.stop_event = threading.Event()

        # Latest annotated frame shared between inference and writer
        self._annotated_frame = None
        self._annotated_lock = threading.Lock()

        self.stats = {
            "fps_capture": 0,
            "fps_processing": 0,
        }

        self.log_interval_sec = 2.0

        # Recognition state
        self.camera_id = cfg["camera"].get("camera_id", "CAM_01")
        self.cooldown_sec = cfg.get("attendance", {}).get("cooldown_sec", 10)
        self._last_event = {}      # employee_id → timestamp

        # Pipeline tuning (new)
        pipe_cfg = cfg.get("pipeline", {})
        self.embed_cooldown_sec = pipe_cfg.get("embed_cooldown_sec", 1.0)
        self.cleanup_interval_sec = pipe_cfg.get("cleanup_interval_sec", 30)
        perf_queue_size = pipe_cfg.get("perf_queue_size", 200)

        # Event CSV Logging State
        event_cfg = cfg.get("event_log", {})
        self.csv_path = event_cfg.get("csv_path", None)
        self.session_timeout_sec = event_cfg.get("timeout_sec", 1200)
        self.session_active = False
        self.session_start = 0.0
        self.session_last_event = 0.0
        self.session_detected_persons = set() # track_ids seen
        self.session_identified_counts = {}   # "employee_id_name" -> count

        # Face size threshold from recognition config
        self.min_face_px = cfg.get("recognition", {}).get("min_face_px", 60)

        # Debug frame buffer — known/ and unknown/ buckets
        dbg_cfg = cfg.get("debug_capture", {})
        if dbg_cfg.get("enabled", False):
            self._debug_buf: "DebugFrameBuffer | None" = DebugFrameBuffer(
                save_dir   = dbg_cfg.get("save_dir", "/tmp/face_debug"),
                max_frames = int(dbg_cfg.get("max_frames", 200)),
            )
        else:
            self._debug_buf: "DebugFrameBuffer | None" = None

        # Debug video writer — rolling MP4 of annotated inference stream
        if dbg_cfg.get("enabled", False) and dbg_cfg.get("video_enabled", False):
            self._video_writer: "DebugVideoWriter | None" = DebugVideoWriter(
                video_dir   = str(Path(dbg_cfg.get("save_dir", "/tmp/face_debug")) / "videos"),
                fps         = float(cfg["camera"]["fps"]),
                width       = int(cfg["camera"]["width"]),
                height      = int(cfg["camera"]["height"]),
                segment_sec = float(dbg_cfg.get("video_segment_sec", 60)),
                max_videos  = int(dbg_cfg.get("max_videos", 10)),
            )
        else:
            self._video_writer: "DebugVideoWriter | None" = None

        # Async DB queues
        self._event_q = queue.Queue()                      # unlimited — never drop
        self._perf_q = queue.Queue(maxsize=perf_queue_size)  # capped — drop oldest

        # Shared accumulator for video write time (updated by _write_loop,
        # read+reset by _infer_loop at each stat snapshot)
        self._video_ms_lock   = threading.Lock()
        self._video_write_sum = 0.0
        self._video_write_cnt = 0

        # Track cleanup timer
        self._last_cleanup = time.time()


    # ------------------------------------------------
    # START / STOP
    # ------------------------------------------------

    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()
        threading.Thread(target=self._write_loop, daemon=True).start()
        threading.Thread(target=self._db_writer_loop, daemon=True).start()
        print("[PIPELINE] Started with embed cooldown "
              f"{self.embed_cooldown_sec}s, "
              f"min face {self.min_face_px}px")

    def stop(self):
        print("[PIPELINE] Stopping...")
        if self.session_active:
            print("[EVENT LOG] Pipeline stopped, flushing active session synchronously.")
            self._write_session_to_csv()
            
        self.stop_event.set()
        self.capture.release()
        if self._video_writer is not None:
            self._video_writer.close()
        if self.stream_manager:
            self.stream_manager.cleanup()

    # ------------------------------------------------
    # CAPTURE LOOP
    # ------------------------------------------------

    def _capture_loop(self):

        frames = 0
        last_time = time.time()

        while not self.stop_event.is_set():

            frame = self.capture.read()
            if frame is None:
                continue

            # Push to infer+write queue — drop oldest if full so we stay on latest frame
            if self.frame_q.full():
                try:
                    self.frame_q.get_nowait()
                except Exception: pass
            self.frame_q.put(frame)

            frames += 1
            now = time.time()

            if now - last_time >= 1:
                self.stats["fps_capture"] = frames
                frames = 0
                last_time = now

    # ------------------------------------------------
    # INFERENCE LOOP
    # Detect on a frame, draw boxes on THAT SAME frame,
    # store the annotated result for _write_loop to pump.
    # ------------------------------------------------

    def _infer_loop(self):

        target_w = self.cfg["camera"]["width"]
        target_h = self.cfg["camera"]["height"]

        frame_count = 0
        stat_start = time.time()

        sum_capture = sum_detect = sum_align = 0
        sum_embed = sum_match = sum_write = sum_total = 0
        sum_face = sum_embed_ok = sum_identified = 0
        sum_image_full = sum_image_raw = sum_face_aligned = sum_face_raw = 0.0
        skipped_small = 0
        skipped_cooldown = 0
        # Last identified employee in this stat window
        last_id_in_window = ""
        last_name_in_window = ""
        last_face_path_in_window = ""

        while not self.stop_event.is_set():

            loop_start = time.time()

            # ---------------- GET FRAME ----------------
            cap_start = time.time()
            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            capture_ms = (time.time() - cap_start) * 1000

            # ---------------- DETECTION ----------------
            det_start = time.time()
            detections = self.detector.detect(frame)
            detect_ms = (time.time() - det_start) * 1000

            face_detected = len(detections) if detections else 0

            align_ms = 0
            embed_ms = 0
            embedding_generated = 0
            active_track_ids = []
            new_detections = []
            # (aligned_patch, raw_face_crop, label) pairs collected AFTER matching
            # so we know whether to route to known/ or unknown/
            _debug_captures: "list[tuple[np.ndarray, np.ndarray, str, str]]" = []

            if detections:
                for det_idx, det in enumerate(detections):
                    track_id = det_idx
                    active_track_ids.append(track_id)

                    x1, y1, x2, y2 = det["bbox"]

                    # Update active session on detection
                    now = time.time()
                    if not self.session_active:
                        self.session_active = True
                        self.session_start = now
                        self.session_detected_persons.clear()
                        self.session_identified_counts.clear()
                    
                    self.session_last_event = now
                    self.session_detected_persons.add(track_id)

                    cfg_rec = self.cfg.get("recognition", {})
                    if "min_face_px" not in cfg_rec:
                        cfg_rec["min_face_px"] = self.min_face_px

                    # Caching logic first, to avoid redundant processing if track is in cooldown
                    now_ts = time.time()
                    cached = self.matcher.get_cached(track_id)

                    if cached is not None:
                        state = self.matcher.track.get(track_id)
                        if (state is not None and (now_ts - state.last_embed_ts) < self.embed_cooldown_sec):
                            label, name, score, top_results = cached
                            skipped_cooldown += 1
                            
                            if label != "unknown":
                                sum_identified += 1

                            new_detections.append({
                                "bbox": (x1, y1, x2, y2),
                                "label": label,
                                "name": name,
                                "score": score,
                                "top_results": top_results,
                            })

                            if label != "unknown":
                                last = self._last_event.get(label, 0)
                                if now_ts - last >= self.cooldown_sec:
                                    self._last_event[label] = now_ts
                                    self._event_q.put((self.camera_id, track_id, label, score))
                            continue

                    a_start = time.time()
                    aligned, face, status = get_final_face_patch(
                        frame,
                        (x1, y1, x2, y2),
                        landmarker=self.landmarker,
                        cfg_recognition=cfg_rec
                    )
                    align_ms += (time.time() - a_start) * 1000

                    if aligned is None:
                        if status == "Too Small":
                            skipped_small += 1
                        else:
                            # logging for why skipped
                            # print(f"[PIPELINE] Skipping face: {status}")
                            pass
                        continue

                    e_start = time.time()
                    emb = self.embedder.embed(aligned)
                    embed_ms += (time.time() - e_start) * 1000

                    if emb is None: continue
                    embedding_generated += 1

                    m_start = time.time()
                    label, name, score, top_results = self.matcher.decide(track_id, emb)
                    match_ms_this = (time.time() - m_start) * 1000
                    sum_match += match_ms_this

                    if label != "unknown":
                        sum_identified += 1
                        # Track last identified for perf log
                        last_id_in_window = label
                        last_name_in_window = name

                    # Collect for debug AFTER we know the label
                    if self._debug_buf is not None:
                        _debug_captures.append((aligned, face, label, name))
                    
                    new_detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "name": name,
                        "score": score,
                        "top_results": top_results,
                    })

                    if label != "unknown":
                        last = self._last_event.get(label, 0)
                        if time.time() - last >= self.cooldown_sec:
                            self._last_event[label] = time.time()
                            self._event_q.put((self.camera_id, track_id, label, score))
                        
                        # Update session ID count
                        id_name_key = f"{label}_{name}"
                        self.session_identified_counts[id_name_key] = self.session_identified_counts.get(id_name_key, 0) + 1

            # -------- PERIODIC TRACK CLEANUP --------
            now = time.time()
            if now - self._last_cleanup >= self.cleanup_interval_sec:
                self.matcher.cleanup_tracks(active_track_ids)
                self._last_cleanup = now

            # -------- DRAW OVERLAYS ON THE SAME FRAME --------
            draw_frame = frame.copy()
            for d in new_detections:
                x1, y1, x2, y2 = d["bbox"]
                label = d["label"]
                name = d["name"]
                score = d["score"]
                top_results = d.get("top_results", [])
                color = (0, 0, 255)   # Red for unknown
                if label != "unknown":
                    color = (0, 255, 0)  # Green for known

                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)

                # ── MAIN LABEL (above bbox) ──────────────────────────────────
                # Always show best DB score so score is visible even for unknown
                if label != "unknown":
                    display_text = f"{name} ({label}) DB:{score:.2f}"
                else:
                    # Show best match score even for unknown so user can see proximity
                    best_score = top_results[0]["score"] if top_results else score
                    display_text = f"unknown  best:{best_score:.2f}"

                y_label = max(y1 - 10, 15)
                cv2.putText(
                    draw_frame, display_text, (x1, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

                # ── TOP-3 CANDIDATES (below bbox) ────────────────────────────
                # Draw below the bbox so they are never clipped by the top edge
                if top_results:
                    frame_h = draw_frame.shape[0]
                    for rank, res in enumerate(top_results):
                        r_name  = res.get("name", "Unknown")
                        r_label = res["label"]
                        r_score = res["score"]
                        r_text  = f"#{rank+1} {r_name} ({r_label}): {r_score:.2f}"
                        y_res   = y2 + 20 * (rank + 1)
                        if y_res < frame_h - 5:
                            cv2.putText(
                                draw_frame, r_text, (x1, y_res),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                            )

            # -------- STORE ANNOTATED FRAME FOR WRITER --------
            with self._annotated_lock:
                self._annotated_frame = draw_frame

            # -------- DEBUG CAPTURE — label-routed (known / unknown) --------
            # Saves each recognised face to known/ and each unrecognised face
            # to unknown/  Only triggered when embedding succeeded (label known).
            # No-face frames are skipped: use the video for continuous coverage.
            if self._debug_buf is not None:
                for patch, face_raw, lbl, pname in _debug_captures:
                    ifull_ms, iraw_ms, falign_ms, fraw_ms = self._debug_buf.push(draw_frame, frame, face_raw, patch, lbl, name=pname)
                    sum_image_full += ifull_ms
                    sum_image_raw += iraw_ms
                    sum_face_aligned += falign_ms
                    sum_face_raw += fraw_ms
                    # Capture the face save path for the last identified person
                    if lbl != "unknown":
                        dbg_dir = Path(self.cfg.get("debug_capture", {}).get("save_dir", "/tmp/face_debug").replace("{runtime}", self.cfg.get("models", {}).get("runtime", "tflite")))
                        bucket_dir = dbg_dir / "known" / f"{lbl}_{pname}"
                        last_face_path_in_window = str(bucket_dir)

            total_ms = (time.time() - loop_start) * 1000

            # ---------------- STATS ----------------
            frame_count += 1
            sum_capture += capture_ms
            sum_detect  += detect_ms
            sum_align   += align_ms
            sum_embed   += embed_ms
            sum_total   += total_ms
            sum_face    += face_detected
            sum_embed_ok += embedding_generated

            now = time.time()
            if now - stat_start >= self.log_interval_sec:
                fps = frame_count / (now - stat_start)
                n   = frame_count

                # Snapshot and reset video write accumulator (thread-safe)
                with self._video_ms_lock:
                    v_sum = self._video_write_sum
                    v_cnt = self._video_write_cnt
                    self._video_write_sum = 0.0
                    self._video_write_cnt = 0
                avg_video_ms = (v_sum / v_cnt) if v_cnt > 0 else 0.0

                perf_data = (
                    fps,
                    sum_capture / n,
                    sum_detect  / n,
                    sum_align   / n,
                    sum_embed   / n,
                    sum_match   / n,
                    0,
                    sum_total   / n,
                    sum_face,
                    sum_embed_ok,
                    sum_image_full / n,
                    sum_image_raw / n,
                    sum_face_aligned / n,
                    sum_face_raw / n,
                    avg_video_ms,
                    sum_identified,
                    last_id_in_window,
                    last_name_in_window,
                    last_face_path_in_window,
                )

                if self._perf_q.full():
                    try: self._perf_q.get_nowait()
                    except queue.Empty: pass
                self._perf_q.put(perf_data)

                self.stats["fps_processing"]  = fps
                self.stats["skipped_small"]   = skipped_small
                self.stats["skipped_cooldown"] = skipped_cooldown

                frame_count  = 0
                stat_start   = now
                sum_capture  = sum_detect = sum_align = 0
                sum_embed    = sum_match = sum_total = sum_face = sum_embed_ok = sum_identified = 0
                sum_image_full = sum_image_raw = sum_face_aligned = sum_face_raw = 0.0
                skipped_small = skipped_cooldown = 0
                last_id_in_window = ""
                last_name_in_window = ""
                last_face_path_in_window = ""

    # ------------------------------------------------
    # WRITE LOOP (constant FPS → FFmpeg never starves)
    # Re-sends the last annotated frame when inference
    # is slower than the target FPS. No bbox drift because
    # we re-use the already-annotated frame, not a raw one.
    # ------------------------------------------------

    def _write_loop(self):

        target_interval = 1.0 / self.cfg["camera"]["fps"]
        last_write = time.time()
        last_frame = None
        last_new_time = time.time()
        stale_warned = False

        while not self.stop_event.is_set():

            # Pace to target FPS
            now = time.time()
            sleep_t = target_interval - (now - last_write)
            if sleep_t > 0:
                time.sleep(sleep_t)

            # Grab latest annotated frame (or re-use last)
            with self._annotated_lock:
                frame = self._annotated_frame

            if frame is not None:
                last_frame = frame
                last_new_time = time.time()
                stale_warned = False
            else:
                frame = last_frame
                stale_sec = time.time() - last_new_time
                if stale_sec > 10 and not stale_warned:
                    logger.warning(
                        f"[WRITER] No annotated frame in {stale_sec:.0f}s — inference stalled?"
                    )
                    stale_warned = True

            if frame is None:
                continue

            last_write = time.time()

            if self.stream_manager:
                try:
                    self.stream_manager.write_frame(frame)
                except Exception:
                    pass

            # Feed the debug video recorder (if enabled) — timed
            if self._video_writer is not None:
                try:
                    _t0 = time.time()
                    self._video_writer.write(frame)
                    _v_ms = (time.time() - _t0) * 1000
                    with self._video_ms_lock:
                        self._video_write_sum += _v_ms
                        self._video_write_cnt  += 1
                except Exception:
                    pass

    # ------------------------------------------------
    # DB WRITER LOOP (new — async queue drain)
    # ------------------------------------------------

    def _db_writer_loop(self):
        """Drain event and perf queues asynchronously.

        Runs in its own thread — inference loop never blocks on DB.
        """

        while not self.stop_event.is_set():

            wrote_something = False

            # --- drain events (attendance logs, never drop) ---
            while not self._event_q.empty():
                try:
                    cam, tid, label, score = self._event_q.get_nowait()
                    asyncio.run_coroutine_threadsafe(
                        self._log_event(cam, tid, label, score),
                        self.loop,
                    )
                    wrote_something = True
                except queue.Empty:
                    break

            # --- drain perf logs ---
            while not self._perf_q.empty():
                try:
                    perf_data = self._perf_q.get_nowait()
                    asyncio.run_coroutine_threadsafe(
                        self._save_perf(*perf_data),
                        self.loop,
                    )
                    wrote_something = True
                except queue.Empty:
                    break

            # --- check session timeout ---
            now = time.time()
            if self.session_active and (now - self.session_last_event) > self.session_timeout_sec:
                print("[EVENT LOG] Session timeout.")
                self._write_session_to_csv()
                wrote_something = True

            # If nothing to write, sleep briefly to avoid busy-spin
            if not wrote_something:
                time.sleep(0.1)

        # --- final flush on stop ---
        if self.session_active:
            print("[EVENT LOG] Pipeline stopped, flushing active session.")
            self._write_session_to_csv()

    def _write_session_to_csv(self):
        if not self.session_active:
            return
        if self.csv_path:
            try:
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
                file_exists = os.path.isfile(self.csv_path)
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['Session Start', 'Session End', 'Total Persons Detected', 'Identified Employee Counts'])
                    
                    start_str = datetime.fromtimestamp(self.session_start).strftime("%Y-%m-%d %H:%M:%S")
                    end_str = datetime.fromtimestamp(self.session_last_event).strftime("%Y-%m-%d %H:%M:%S")
                    persons_count = len(self.session_detected_persons)
                    counts_json = json.dumps(self.session_identified_counts)
                    
                    writer.writerow([start_str, end_str, persons_count, counts_json])
                print(f"[EVENT LOG] Written stats to {self.csv_path}")
            except Exception as e:
                print(f"[EVENT LOG ERROR] Could not write to CSV: {e}")
        
        # Reset session
        self.session_active = False
        self.session_detected_persons.clear()
        self.session_identified_counts.clear()

    # ------------------------------------------------
    # LOG EVENT
    # ------------------------------------------------

    async def _log_event(self, camera_id, track_id, result, score):
        try:
            async with self.db.begin() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO events (ts, camera_id, track_id, result, score)
                        VALUES (:ts, :cam, :tid, :res, :score)
                    """),
                    {
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "cam": camera_id,
                        "tid": track_id,
                        "res": result,
                        "score": float(score),
                    },
                )
            print(f"[EVENT] {result} (score={score:.2f})")
        except Exception as e:
            print(f"[EVENT ERROR] {e}")

    # ------------------------------------------------
    # SAVE PERFORMANCE
    # ------------------------------------------------

    async def _save_perf(
        self,
        fps,
        capture_ms,
        detect_ms,
        align_ms,
        embed_ms,
        match_ms,
        write_ms,
        total_ms,
        face_detected,
        embedding_generated,
        image_full_save_ms,
        image_raw_save_ms,
        face_aligned_save_ms,
        face_raw_save_ms,
        video_write_ms,
        faces_identified,
        last_identified_id="",
        last_identified_name="",
        last_face_save_path="",
    ):

        try:
            async with self.db.begin() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO performance_logs (
                            ts, fps,
                            capture_ms, detect_ms, align_ms,
                            embed_ms, match_ms, write_ms, total_ms,
                            face_detected, embedding_generated,
                            image_full_save_ms, image_raw_save_ms,
                            face_aligned_save_ms, face_raw_save_ms,
                            debug_frame_ms, debug_face_ms,
                            video_write_ms, faces_identified,
                            last_identified_id, last_identified_name,
                            last_face_save_path
                        )
                        VALUES (
                            :ts, :fps,
                            :capture_ms, :detect_ms, :align_ms,
                            :embed_ms, :match_ms, :write_ms, :total_ms,
                            :face_detected, :embedding_generated,
                            :image_full_save_ms, :image_raw_save_ms,
                            :face_aligned_save_ms, :face_raw_save_ms,
                            0, 0,
                            :video_write_ms, :faces_identified,
                            :last_identified_id, :last_identified_name,
                            :last_face_save_path
                        )
                    """),
                    {
                        "ts":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "fps":              float(fps),
                        "capture_ms":       float(capture_ms),
                        "detect_ms":        float(detect_ms),
                        "align_ms":         float(align_ms),
                        "embed_ms":         float(embed_ms),
                        "match_ms":         float(match_ms),
                        "write_ms":         float(write_ms),
                        "total_ms":         float(total_ms),
                        "face_detected":    int(face_detected),
                        "embedding_generated": int(embedding_generated),
                        "image_full_save_ms": float(image_full_save_ms),
                        "image_raw_save_ms": float(image_raw_save_ms),
                        "face_aligned_save_ms": float(face_aligned_save_ms),
                        "face_raw_save_ms": float(face_raw_save_ms),
                        "video_write_ms":   float(video_write_ms),
                        "faces_identified": int(faces_identified),
                        "last_identified_id":   str(last_identified_id),
                        "last_identified_name": str(last_identified_name),
                        "last_face_save_path":  str(last_face_save_path),
                    },
                )
        except Exception as e:
            print(f"[PERF LOG ERROR] {e}")