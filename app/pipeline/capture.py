import cv2
import time
import threading
import requests
import numpy as np
import logging

logger = logging.getLogger(__name__)

# At 1280x720 MJPEG, individual frames can be 150–400 KB.
# Old 5 MB cap was too aggressive — resetting the whole buffer discards
# partial frame data and makes the stream appear frozen.
# Raised to 20 MB; on overflow we trim to the last JPEG start marker
# instead of nuking everything.
MAX_BUFFER_SIZE = 20 * 1024 * 1024  # 20 MB safety cap


class OpenCVCapture:

    def __init__(self, source=0, width=640, height=480, fps=15):

        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        # Track when we last got a fresh frame
        self.last_frame_time = time.time()

        if isinstance(self.source, str) and self.source.startswith("http"):
            print("[CAPTURE] Opening MJPEG stream...")
            self.thread = threading.Thread(target=self._mjpeg_loop, daemon=True)
            self.thread.start()
        else:
            print("[CAPTURE] Opening local camera...")
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    # ------------------------------------------------
    # MJPEG LOOP (ROBUST with read timeout)
    # ------------------------------------------------

    def _mjpeg_loop(self):

        while self.running:

            try:
                # timeout=(connect_timeout, read_timeout)
                # read_timeout ensures iter_content doesn't hang forever
                stream = requests.get(
                    self.source,
                    stream=True,
                    timeout=(5, 10)
                )
                bytes_buffer = b""

                # Use 16 KB chunks — faster receive for large 720p MJPEG frames
                for chunk in stream.iter_content(chunk_size=16384):

                    if not self.running:
                        break

                    bytes_buffer += chunk

                    # Safety: cap buffer to prevent unbounded memory growth.
                    # Instead of nuking everything, trim to the last JPEG start
                    # marker so we don't discard a frame already in progress.
                    if len(bytes_buffer) > MAX_BUFFER_SIZE:
                        last_start = bytes_buffer.rfind(b'\xff\xd8')
                        if last_start > 0:
                            logger.warning(
                                f"[CAPTURE] Buffer too large ({len(bytes_buffer)} bytes)"
                                f" — trimming to last JPEG start at offset {last_start}"
                            )
                            bytes_buffer = bytes_buffer[last_start:]
                        else:
                            # No start marker found at all — full reset
                            logger.warning(
                                f"[CAPTURE] Buffer overflow ({len(bytes_buffer)} bytes)"
                                f" — no JPEG start found, full reset"
                            )
                            bytes_buffer = b""
                        continue

                    # Extract ALL complete JPEGs in the current buffer
                    while True:
                        start = bytes_buffer.find(b'\xff\xd8')
                        if start == -1:
                            break
                        
                        next_start = bytes_buffer.find(b'\xff\xd8', start + 2)
                        if next_start == -1:
                            # Wait for the next frame's start to ensure we have the FULL current frame.
                            # This bypasses any fake EOI markers inside embedded EXIF thumbnails.
                            break
                            
                        # Find the TRUE end of the current frame (the LAST EOI before the next frame)
                        end = bytes_buffer.rfind(b'\xff\xd9', start, next_start)
                        
                        if end != -1:
                            jpg = bytes_buffer[start:end + 2]
                            # Keep everything from the next frame onward
                            bytes_buffer = bytes_buffer[next_start:]
                            
                            if len(jpg) == 0:
                                continue
                                
                            # Frame-drop logic: if we have another full frame ready in the buffer,
                            # skip this one to instantly catch up to the real-time feed.
                            future_next = bytes_buffer.find(b'\xff\xd8', 2)
                            if future_next != -1:
                                future_end = bytes_buffer.rfind(b'\xff\xd9', 0, future_next)
                                if future_end != -1:
                                    continue

                            frame = cv2.imdecode(
                                np.frombuffer(jpg, dtype=np.uint8),
                                cv2.IMREAD_COLOR
                            )

                            if frame is None:
                                continue

                            # Only resize if needed
                            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                                frame = cv2.resize(
                                    frame,
                                    (self.width, self.height)
                                )

                            with self.lock:
                                self.frame = frame
                                self.last_frame_time = time.time()
                        else:
                            break

            except requests.exceptions.ReadTimeout:
                logger.warning(
                    "[CAPTURE] MJPEG read timeout — reconnecting..."
                )
                time.sleep(1)

            except requests.exceptions.ConnectionError:
                logger.warning(
                    "[CAPTURE] MJPEG connection lost — retrying in 2s..."
                )
                time.sleep(2)

            except Exception as e:
                logger.error(f"[CAPTURE] MJPEG error: {e}")
                time.sleep(1)

    # ------------------------------------------------
    # READ
    # ------------------------------------------------

    def read(self):

        if isinstance(self.source, str) and self.source.startswith("http"):
            with self.lock:
                return self.frame.copy() if self.frame is not None else None

        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            # Resize to the configured target size if the webcam returned a
            # different resolution (many cameras silently ignore CAP_PROP_FRAME_*).
            # FFmpeg expects every frame to be exactly width×height bytes.
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            return frame

    # ------------------------------------------------
    # STALE CHECK
    # ------------------------------------------------

    def is_stale(self, threshold_sec=10):
        """Return True if no fresh frame in `threshold_sec` seconds."""
        return (time.time() - self.last_frame_time) > threshold_sec

    # ------------------------------------------------
    # RELEASE
    # ------------------------------------------------

    def release(self):

        self.running = False

        if hasattr(self, "cap"):
            self.cap.release()
