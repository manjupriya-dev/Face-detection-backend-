import subprocess
from pathlib import Path
import logging
import threading
import time
import os
import signal
import uuid

logger = logging.getLogger(__name__)

# Timeout for FFmpeg stdin write (seconds)
WRITE_TIMEOUT_SEC = 3


class StreamManager:

    def __init__(self, config):
        self.config = config
        self.hls_output_dir = Path(
            config["streaming"]["hls_output_dir"]
        )

        self.process = None
        self.running = False
        self.monitor_thread = None
        self.session_id = None

        # Write health tracking
        self._last_write_ok = time.time()
        self._write_lock = threading.Lock()
        self._restarting = False

    # ---------------------------------------------------
    # START PIPE ENCODER (LIVE EDGE SAFE VERSION)
    # ---------------------------------------------------

    def start_pipe_encoder(self):

        camera_cfg = self.config["camera"]

        width = camera_cfg["width"]
        height = camera_cfg["height"]

        # Must match real processing FPS
        fps = camera_cfg["fps"]

        self.hls_output_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_hls_files()

        playlist = self.hls_output_dir / "playlist.m3u8"
        segment_pattern = self.hls_output_dir / "segment_%d.ts"

        cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "warning",
                "-y",

                # ---------- INPUT ----------
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",

                # ---------- ENCODER ----------
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
                "-profile:v", "baseline",
                "-level", "3.1",
                "-crf", "23",

                # Proper GOP (2s keyframe interval)
                "-g", str(fps * 2),
                "-keyint_min", str(fps * 2),
                "-sc_threshold", "0",

                # Modern frame pacing (replaces -vsync)
                "-fps_mode", "cfr",

                # ---------- HLS ----------
                "-f", "hls",
                "-hls_time", "1",
                "-hls_list_size", "3",

                # Clean, non-conflicting flags
                "-hls_flags", "delete_segments+independent_segments+program_date_time",
                "-hls_allow_cache", "0",
                "-hls_start_number_source", "epoch",

                "-hls_segment_filename", str(segment_pattern),
                str(playlist),
            ]

        logger.info("[FFMPEG] Starting stable encoder")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        self.running = True
        self._last_write_ok = time.time()
        self._restarting = False

        self.monitor_thread = threading.Thread(
            target=self._monitor_process,
            daemon=True
        )
        self.monitor_thread.start()

        # Stderr reader thread — prevents pipe buffer filling
        threading.Thread(
            target=self._stderr_reader,
            daemon=True
        ).start()


    # ---------------------------------------------------
    # WRITE FRAME (with timeout protection)
    # ---------------------------------------------------

    def write_frame(self, frame):

        if not self.running:
            return

        if not self.process or not self.process.stdin:
            return

        # Use a lock so only one thread writes at a time
        acquired = self._write_lock.acquire(timeout=WRITE_TIMEOUT_SEC)
        if not acquired:
            logger.error("[FFMPEG] Write lock timeout — pipe likely blocked")
            self._trigger_restart()
            return

        try:
            data = frame.tobytes()

            # Write in a separate thread with timeout to detect blocking
            write_done = threading.Event()
            write_error = [None]

            def _do_write():
                try:
                    self.process.stdin.write(data)
                    self.process.stdin.flush()
                except Exception as e:
                    write_error[0] = e

                write_done.set()

            t = threading.Thread(target=_do_write, daemon=True)
            t.start()

            if not write_done.wait(timeout=WRITE_TIMEOUT_SEC):
                logger.error(
                    "[FFMPEG] stdin.write() blocked for "
                    f"{WRITE_TIMEOUT_SEC}s — restarting encoder"
                )
                self._trigger_restart()
                return

            if write_error[0] is not None:
                raise write_error[0]

            self._last_write_ok = time.time()

        except BrokenPipeError:
            logger.error("[FFMPEG] Broken pipe")
            self._trigger_restart()
        except Exception as e:
            logger.error(f"[FFMPEG] Write failed: {e}")
        finally:
            self._write_lock.release()

    # ---------------------------------------------------
    # STDERR READER (prevent pipe buffer fill)
    # ---------------------------------------------------

    def _stderr_reader(self):
        """Read FFmpeg stderr in background so the pipe doesn't block."""
        try:
            while self.running and self.process and self.process.stderr:
                line = self.process.stderr.readline()
                if not line:
                    break
                msg = line.decode("utf-8", errors="replace").strip()
                if msg:
                    logger.warning(f"[FFMPEG STDERR] {msg}")
        except Exception:
            pass

    # ---------------------------------------------------
    # MONITOR (also checks write health)
    # ---------------------------------------------------

    def _monitor_process(self):

        while self.running:

            if self.process is None:
                break

            ret = self.process.poll()

            if ret is not None:
                logger.error(f"[FFMPEG] Crashed with code {ret}")
                self._trigger_restart()
                break

            # Check if writes have stalled (no successful write in 10s)
            if time.time() - self._last_write_ok > 10:
                logger.error(
                    "[FFMPEG] No successful write in 10s — restarting"
                )
                self._trigger_restart()
                break

            time.sleep(2)

    # ---------------------------------------------------
    # RESTART (thread-safe, prevents double-restart)
    # ---------------------------------------------------

    def _trigger_restart(self):
        """Thread-safe restart — only one restart at a time."""
        if self._restarting:
            return
        self._restarting = True

        threading.Thread(
            target=self._restart,
            daemon=True
        ).start()

    def _restart(self):

        logger.warning("[FFMPEG] Restarting encoder...")

        self.cleanup()
        time.sleep(1)
        self.start_pipe_encoder()

    # ---------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------

    def _cleanup_hls_files(self):

        if self.hls_output_dir.exists():
            for file in self.hls_output_dir.glob("*"):
                try:
                    file.unlink()
                except Exception:
                    pass

    def cleanup(self):

        self.running = False

        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
            except Exception:
                pass

            try:
                if self.process.stderr:
                    self.process.stderr.close()
            except Exception:
                pass

            try:
                os.kill(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

        self.process = None
        self._cleanup_hls_files()
