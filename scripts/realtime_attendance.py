import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import torch

from core.yolo_face_detector import YOLOFaceDetector
from core.facenet_embedder import FaceNetEmbedder
from core.matcher import EmbeddingMatcher
from core.liveness_blink import BlinkLiveness
from core.attendance_writer import AttendanceWriter


def main():
    # ---------------- CONFIG ----------------
    YOLO_PATH = "models/yolov8n-face-lindevs.pt"  # đổi nếu bạn rename khác
    CONF_THRES = 0.5

    MATCH_THRESHOLD = 0.55       # giảm xuống 0.52 nếu bạn có ít ảnh
    COOLDOWN_SEC = 600           # 10 phút
    LIVENESS_TIMEOUT = 3.0       # 3 giây để blink
    # ----------------------------------------


    # 1) Init modules
    detector = YOLOFaceDetector(model_path=YOLO_PATH, conf=CONF_THRES)
    embedder = FaceNetEmbedder()
    matcher = EmbeddingMatcher("data/embeddings", threshold=MATCH_THRESHOLD)
    liveness = BlinkLiveness(timeout=LIVENESS_TIMEOUT)
    writer = AttendanceWriter("data/attendance", cooldown_sec=COOLDOWN_SEC)

    if len(matcher.gallery) == 0:
        print("[ERROR] No embeddings loaded! Please run build_embeddings.py first.")
        return

    # 2) Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("\n=== Realtime Attendance Started ===")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        # Nếu không có face => reset liveness
        if len(faces) == 0:
            liveness.reset()
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            # Lấy khuôn mặt lớn nhất (chỉ xử lý 1 người / 1 thời điểm cho đơn giản)
            faces.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
            box = faces[0]

            face_img = detector.crop_face(frame, box, margin=10)
            if face_img is not None:

                # Liveness
                live, info = liveness.update(face_img)

                # Draw bounding box
                cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 2)
                cv2.putText(frame, info, (box["x1"], box["y1"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if live:
                    # FaceNet embed + match
                    emb = embedder.embed(face_img)
                    sid, score = matcher.match(emb)

                    if sid is not None:
                        ok, msg = writer.mark(sid, name=f"Student_{sid}")
                        cv2.putText(frame, f"ID={sid} score={score:.2f}", (box["x1"], box["y2"] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame, msg, (box["x1"], box["y2"] + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, f"Unknown score={score:.2f}", (box["x1"], box["y2"] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Reset challenge sau khi đã xử lý
                    liveness.reset()

        cv2.imshow("Attendance YOLOv8 + FaceNet + Blink Liveness", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
