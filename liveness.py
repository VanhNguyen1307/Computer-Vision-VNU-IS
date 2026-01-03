import cv2
import mediapipe as mp
import time
import math

mp_face_mesh = mp.solutions.face_mesh

# Các mốc mắt (theo FaceMesh) – mình lấy 6 điểm cho 2 mắt
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_idx, img_w, img_h):
    points = []
    for i in eye_idx:
        lm = landmarks[i]
        points.append((int(lm.x * img_w), int(lm.y * img_h)))

    # 6 điểm: p0..p5
    p0, p1, p2, p3, p4, p5 = points

    def dist(a, b):
        return math.dist(a, b)

    # công thức EAR đơn giản
    vertical = dist(p1, p5) + dist(p2, p4)
    horizontal = dist(p0, p3)
    ear = vertical / (2.0 * horizontal + 1e-6)
    return ear

def check_blink_liveness(cap, timeout=5, need_blinks=2):
    """Yêu cầu người dùng nháy mắt need_blinks lần trong timeout giây."""
    start_time = time.time()
    blink_count = 0
    blink_state = False  # đang nhắm hay mở

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                lm = results.multi_face_landmarks[0].landmark

                left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                # ngưỡng EAR: < 0.2 coi như nhắm mắt (tùy chỉnh)
                if ear < 0.20:
                    if not blink_state:
                        blink_state = True
                else:
                    if blink_state:
                        blink_state = False
                        blink_count += 1

            # hiển thị hướng dẫn
            cv2.putText(frame,
                        f"Please blink {need_blinks} times. Detected: {blink_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.imshow("Liveness check - blink", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if blink_count >= need_blinks:
                cv2.destroyWindow("Liveness check - blink")
                return True

    cv2.destroyWindow("Liveness check - blink")
    return False
