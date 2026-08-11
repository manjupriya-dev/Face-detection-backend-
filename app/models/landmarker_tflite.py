import cv2
import numpy as np
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



class FaceLandmarkerWrapper:

    def __init__(self, model_path, use_gpu: bool = True):

        # MediaPipe on Jetson Orin is built WITHOUT GPU/GL support
        # (ImageCloneCalculator: GPU processing is disabled in build flags).
        # Attempting Delegate.GPU raises NotImplementedError at runtime, so we
        # unconditionally use CPU here regardless of the use_gpu config flag.
        # UNet detector and ArcFace embedder use their own GPU paths independently.
        if use_gpu:
            print(
                "[MEDIAPIPE LANDMARKER] Note: GPU disabled in this mediapipe build — "
                "using CPU (landmark quality is unaffected)"
            )
        else:
            print("[MEDIAPIPE LANDMARKER] CPU")

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU,
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()

        # Eye landmark indices (MediaPipe 478-point mesh)
        self.LEFT_EYE_ID  = 33
        self.RIGHT_EYE_ID = 263

        # Target size expected by the ArcFace model
        self.target_size = (112, 112)

    # ------------------------------------------------------------
    # ALIGN FACE
    # ------------------------------------------------------------

    def align(self, face_bgr):
        """
        Alignment matching the custom ArcFace training pipeline:
        - Rotates to level the eyes
        - Scales so eye distance = 0.30 * target_width
        - Places eye midpoint at (0.5w, 0.4h)
        """
        if face_bgr is None:
            return None

        # Convert BGR → RGB (MediaPipe expects RGB)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=face_rgb,
        )

        with self._lock:
            result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w, _ = face_bgr.shape

        left_eye  = np.array([landmarks[self.LEFT_EYE_ID].x  * w,
                               landmarks[self.LEFT_EYE_ID].y  * h])
        right_eye = np.array([landmarks[self.RIGHT_EYE_ID].x * w,
                               landmarks[self.RIGHT_EYE_ID].y * h])

        dY    = right_eye[1] - left_eye[1]
        dX    = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        dist  = np.sqrt(dX**2 + dY**2)
        scale = (0.30 * self.target_size[0]) / (dist + 1e-6)

        center = ((left_eye + right_eye) / 2).astype(np.float32)
        M = cv2.getRotationMatrix2D(tuple(center), angle, scale)

        # Translate so eye midpoint lands at (0.5w, 0.4h)
        M[0, 2] += self.target_size[0] * 0.5 - center[0]
        M[1, 2] += self.target_size[1] * 0.4 - center[1]

        aligned = cv2.warpAffine(
            face_bgr,
            M,
            self.target_size,
            flags=cv2.INTER_CUBIC,
        )

        return aligned
