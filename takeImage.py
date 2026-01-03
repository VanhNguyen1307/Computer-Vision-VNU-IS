import csv
import os
import cv2
import numpy as np
import time

# ✅ NEW: pipeline modules
from core.yolo_face_detector import YOLOFaceDetector
from core.facenet_embedder import FaceNetEmbedder


def TakeImage(l1, l2, haarcasecade_path, trainimage_path, message, err_screen, text_to_speech):
    """
    Register new student:
    - Capture images
    - Save into TrainingImage/<Enrollment>_<Name>/
    - Save Enrollment + Name to StudentDetails/studentdetails.csv
    - ✅ Auto build FaceNet embedding and save to data/embeddings/<Enrollment>.npy
    """

    if (l1 == "") and (l2 == ""):
        t = "Please Enter the Enrollment Number and Name."
        text_to_speech(t)
        return
    elif l1 == '':
        t = "Please Enter the Enrollment Number."
        text_to_speech(t)
        return
    elif l2 == "":
        t = "Please Enter the Name."
        text_to_speech(t)
        return

    Enrollment = str(l1).strip()
    Name = str(l2).strip()

    try:
        # ============ PREPARE DIR ============
        directory = f"{Enrollment}_{Name}"
        path = os.path.join(trainimage_path, directory)

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            # folder exists -> still allow adding more images
            pass

        # ============ OPEN CAMERA ============
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(0.5)

        if not cam.isOpened():
            text_to_speech("Cannot open camera. Close other camera apps.")
            return

        sampleNum = 0
        max_samples = 15  # ✅ bạn có thể đổi 15-30 tùy ý

        text_to_speech("Camera opened. Look at the camera. Press Q to stop.")

        while True:
            ret, img = cam.read()
            if not ret or img is None:
                continue

            show = img.copy()
            cv2.putText(show, f"Capturing: {sampleNum}/{max_samples}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Register Face (Press Q to stop)", show)

            key = cv2.waitKey(1) & 0xFF

            # capture every few frames
            if sampleNum < max_samples:
                # save frame every ~0.2s
                if key != ord("q"):
                    sampleNum += 1
                    img_path = os.path.join(path, f"{Name}_{Enrollment}_{sampleNum}.jpg")
                    cv2.imwrite(img_path, img)

            if key == ord("q"):
                break
            if sampleNum >= max_samples:
                break

        cam.release()
        cv2.destroyAllWindows()

        # ============ SAVE STUDENT DETAIL ============
        os.makedirs("StudentDetails", exist_ok=True)
        student_csv = "StudentDetails/studentdetails.csv"

        # nếu file chưa có header -> thêm header
        if not os.path.exists(student_csv):
            with open(student_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Enrollment", "Name"])

        # thêm dòng Enrollment, Name nếu chưa tồn tại
        existing = set()
        try:
            with open(student_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 1:
                        existing.add(str(row[0]).strip())
        except:
            pass

        if Enrollment not in existing:
            with open(student_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([Enrollment, Name])

        # ============ ✅ AUTO BUILD EMBEDDING ============
        message.configure(text="Building embeddings...")
        text_to_speech("Building embedding. Please wait.")

        detector = YOLOFaceDetector(model_path="models/yolov8n-face-lindevs.pt", conf=0.5)
        embedder = FaceNetEmbedder()

        embeddings = []

        for file in os.listdir(path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_file = os.path.join(path, file)
            img = cv2.imread(img_file)

            if img is None:
                continue

            faces = detector.detect(img)
            if len(faces) == 0:
                continue

            # choose largest face
            faces.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
            box = faces[0]
            face_img = detector.crop_face(img, box, margin=10)

            if face_img is None or face_img.size == 0:
                continue

            emb = embedder.embed(face_img)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) == 0:
            t = "No face embeddings could be created. Please register again with clearer face."
            text_to_speech(t)
            message.configure(text=t)
            return

        # average embeddings (robust)
        emb_avg = np.mean(np.stack(embeddings), axis=0)

        os.makedirs("data/embeddings", exist_ok=True)
        save_path = os.path.join("data/embeddings", f"{Enrollment}.npy")
        np.save(save_path, emb_avg)

        res = f"Registered successfully: {Enrollment} - {Name} (embedding saved)"
        message.configure(text=res)
        text_to_speech(res)

    except Exception as ex:
        text_to_speech(f"Error: {ex}")
        message.configure(text=f"Error: {ex}")
