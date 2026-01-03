import time
import numpy as np
import cv2
import mediapipe as mp


class BlinkLiveness:
    """
    Challenge-based liveness by blink counting.
    Uses MediaPipe FaceMesh to get eye landmarks and compute EAR.

    update(face_img, need_blinks=2) -> (live: bool, info: str)

    live == True when blink_count >= need_blinks within timeout seconds.
    """

    # FaceMesh landmark indices for eyes (MediaPipe)
    # Left eye (user's left) & Right eye
    # We'll use these 6 points per eye to compute EAR
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, timeout=8.0, ear_thresh=0.25, consec_frames=2, max_missing=10):
        self.timeout = float(timeout)
        self.ear_thresh = float(ear_thresh)
        self.consec_frames = int(consec_frames)
        self.max_missing = int(max_missing)

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,        # improves eye landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.blink_count = 0
        self.closed_frames = 0
        self.missing_frames = 0
        self.prev_eye_closed = False

    # ---------------- EAR CALC ----------------
    @staticmethod
    def _dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def _ear_from_landmarks(self, lm, eye_idx):
        """
        EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
        eye_idx = list of 6 landmark indices
        """
        p1 = lm[eye_idx[0]]
        p2 = lm[eye_idx[1]]
        p3 = lm[eye_idx[2]]
        p4 = lm[eye_idx[3]]
        p5 = lm[eye_idx[4]]
        p6 = lm[eye_idx[5]]

        vertical1 = self._dist(p2, p6)
        vertical2 = self._dist(p3, p5)
        horizontal = self._dist(p1, p4)

        if horizontal < 1e-6:
            return 0.0
        return (vertical1 + vertical2) / (2.0 * horizontal)

    def _extract_landmarks(self, results, w, h):
        """
        Convert MediaPipe normalized landmarks to pixel coordinates.
        Support both 468 (default) and 478 (refine_landmarks=True) landmarks.
        """
        face_landmarks = results.multi_face_landmarks[0]
        n = len(face_landmarks.landmark)  # ✅ dynamic length (468 or 478)

        lm = np.zeros((n, 2), dtype=np.float32)
        for i, pt in enumerate(face_landmarks.landmark):
            lm[i] = np.array([pt.x * w, pt.y * h], dtype=np.float32)

        return lm


    # ---------------- MAIN UPDATE ----------------
    def update(self, face_img, need_blinks=2):
        """
        face_img: cropped face image (BGR or RGB). We'll auto-handle.
        need_blinks: number of blinks required to pass.

        returns (live, info_string)
        """
        now = time.time()
        elapsed = now - self.start_time

        # timeout
        if elapsed > self.timeout:
            self.reset()
            return False, "Liveness timeout - try again"

        if face_img is None or face_img.size == 0:
            self.missing_frames += 1
            if self.missing_frames > self.max_missing:
                self.reset()
                return False, "No face - reset"
            return False, f"Hold still... {self.blink_count}/{need_blinks}"

        h, w = face_img.shape[:2]
        if w < 80 or h < 80:
            return False, f"Move closer... {self.blink_count}/{need_blinks}"

        # Ensure RGB for mediapipe
        # If it's BGR (OpenCV default), convert.
        # Heuristic: assume it's BGR unless it's already RGB passed by user.
        # We'll just always convert BGR->RGB (safe in most cases).
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            self.missing_frames += 1
            if self.missing_frames > self.max_missing:
                self.reset()
                return False, "Face lost - reset"
            return False, f"Detecting face... {self.blink_count}/{need_blinks}"

        self.missing_frames = 0  # reset missing counter

        lm = self._extract_landmarks(results, w, h)

        left_ear = self._ear_from_landmarks(lm, self.LEFT_EYE)
        right_ear = self._ear_from_landmarks(lm, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

        # Eye closed?
        eye_closed = ear < self.ear_thresh

        # count blink: transition from closed -> open
        if eye_closed:
            self.closed_frames += 1
            self.prev_eye_closed = True
        else:
            # if previously closed long enough -> blink detected
            if self.prev_eye_closed and self.closed_frames >= self.consec_frames:
                self.blink_count += 1
            self.closed_frames = 0
            self.prev_eye_closed = False

        # liveness passed?
        if self.blink_count >= need_blinks:
            self.reset()
            return True, "Liveness passed ✅"

        remain = int(max(0, self.timeout - elapsed))
        info = f"Blink {need_blinks} times | {self.blink_count}/{need_blinks} | EAR={ear:.2f} | {remain}s"
        return False, info
