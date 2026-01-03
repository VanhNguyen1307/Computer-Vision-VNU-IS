import tkinter as tk
from tkinter import *
import os
import cv2
import csv
import pandas as pd
import datetime
import time
import random

# ==== YOUR NEW PIPELINE MODULES ====
from core.yolo_face_detector import YOLOFaceDetector
from core.facenet_embedder import FaceNetEmbedder
from core.matcher import EmbeddingMatcher
from core.liveness_blink import BlinkLiveness
from core.attendance_writer import AttendanceWriter


studentdetail_path = "./StudentDetails/studentdetails.csv"
attendance_path = "Attendance"


def load_student_map(csv_path):
    """
    Load Enrollment->Name mapping from StudentDetails/studentdetails.csv
    """
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "Enrollment" not in df.columns or "Name" not in df.columns:
            return {}
        mp = {}
        for _, row in df.iterrows():
            mp[str(row["Enrollment"]).strip()] = str(row["Name"]).strip()
        return mp
    except:
        return {}


def subjectChoose(text_to_speech):

    def FillAttendance():
        print(">>> FillAttendance clicked")

        sub = tx.get().strip()
        now = time.time()
        future = now + 20  # giống bài gốc chạy 20s

        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
            return

        # ============ LOAD NAME MAP ============
        student_map = load_student_map(studentdetail_path)

        try:
            # ============ INIT PIPELINE ============
            detector = YOLOFaceDetector(model_path="models/yolov8n-face-lindevs.pt", conf=0.5)
            embedder = FaceNetEmbedder()
            matcher = EmbeddingMatcher(embeddings_dir="data/embeddings", threshold=0.55)

            # ✅ Challenge-based blink liveness
            liveness = BlinkLiveness(timeout=10.0, ear_thresh=0.25, consec_frames=2)
            need_blinks = random.choice([1, 2, 3])

            writer = AttendanceWriter(attendance_dir="Attendance", cooldown_sec=600)

            if len(matcher.gallery) == 0:
                msg = "No embeddings found. Please run: python scripts/build_embeddings.py"
                Notifica.configure(text=msg, bg="black", fg="yellow", width=60, font=("times", 14, "bold"))
                Notifica.place(x=20, y=250)
                text_to_speech(msg)
                return

            # ============ OPEN CAMERA (FIX GRAY SCREEN) ============
            print(">>> Init pipeline done, opening camera...")
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            time.sleep(0.5)

            if not cam.isOpened():
                raise RuntimeError("Cannot open camera. Close other camera apps.")

            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cam.set(cv2.CAP_PROP_FPS, 30)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = cam.read()
            print(">>> Camera first read ret =", ret, "| frame None?", frame is None)
            if not ret or frame is None:
                raise RuntimeError("Camera opened but cannot read frames. Try CAP_MSMF or close other apps.")

            cv2.namedWindow("Filling Attendance...", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Filling Attendance...", 900, 650)
            font_cv = cv2.FONT_HERSHEY_SIMPLEX

            col_names = ["Enrollment", "Name"]
            attendance = pd.DataFrame(columns=col_names)

            verified_ids = set()
            current_id = None

            path = os.path.join(attendance_path, sub)
            os.makedirs(path, exist_ok=True)

            # ============ MAIN LOOP ============
            while True:
                ret, im = cam.read()
                if not ret or im is None:
                    print(">>> cam.read() failed")
                    break

                faces = detector.detect(im)

                if len(faces) == 0:
                    current_id = None
                    liveness.reset()
                    cv2.putText(im, "No face detected", (20, 40), font_cv, 1, (0, 0, 255), 2)

                else:
                    faces.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
                    box = faces[0]
                    face_img = detector.crop_face(im, box, margin=10)

                    cv2.rectangle(im, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 2)

                    if face_img is not None and face_img.size != 0:
                        emb = embedder.embed(face_img)
                        sid, score = matcher.match(emb)

                        if sid is None:
                            current_id = None
                            liveness.reset()
                            cv2.putText(im, f"Unknown ({score:.2f})",
                                        (box["x1"], box["y1"] - 10),
                                        font_cv, 0.8, (0, 0, 255), 2)

                        else:
                            sid_str = str(sid)
                            name = student_map.get(sid_str, f"Student_{sid_str}")

                            # ---------------- LIVENESS CHECK ----------------
                            if sid_str not in verified_ids:

                                # nếu đổi người thì reset liveness
                                if current_id != sid_str:
                                    current_id = sid_str
                                    liveness.reset()

                                live, info = liveness.update(face_img, need_blinks=need_blinks)

                                cv2.putText(im, f"{sid_str} - {name} ({score:.2f})",
                                            (box["x1"], box["y1"] - 50),
                                            font_cv, 0.8, (255, 255, 0), 2)

                                cv2.putText(im, f"Challenge: Blink {need_blinks} times",
                                            (box["x1"], box["y1"] - 25),
                                            font_cv, 0.8, (0, 255, 255), 2)

                                cv2.putText(im, info,
                                            (box["x1"], box["y1"] - 5),
                                            font_cv, 0.7, (0, 255, 255), 2)

                                if live:
                                    verified_ids.add(sid_str)

                                    if sid_str not in attendance["Enrollment"].values:
                                        attendance.loc[len(attendance)] = [sid_str, name]

                                    ok, msg = writer.mark(sid_str, name=name)

                                    cv2.putText(im, msg,
                                                (box["x1"], box["y2"] + 30),
                                                font_cv, 0.8, (0, 255, 0), 2)

                                    liveness.reset()

                            else:
                                # already verified
                                cv2.putText(im, f"Verified: {sid_str} - {name}",
                                            (box["x1"], box["y1"] - 10),
                                            font_cv, 0.8, (0, 255, 0), 2)

                                if sid_str not in attendance["Enrollment"].values:
                                    attendance.loc[len(attendance)] = [sid_str, name]

                cv2.imshow("Filling Attendance...", im)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if time.time() > future:
                    break

            # ============ SAVE CSV ============
            attendance = attendance.drop_duplicates(["Enrollment"], keep="first")

            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            Hour, Minute, Second = timeStamp.split(":")

            attendance[date] = 1

            fileName = (
                f"{path}/"
                + sub
                + "_"
                + date
                + "_"
                + Hour
                + "-"
                + Minute
                + "-"
                + Second
                + ".csv"
            )

            attendance.to_csv(fileName, index=False)

            m = "Attendance Filled Successfully of " + sub
            Notifica.configure(text=m, bg="black", fg="yellow", width=40,
                               relief=RIDGE, bd=5, font=("times", 15, "bold"))
            text_to_speech(m)
            Notifica.place(x=20, y=250)

            cam.release()
            cv2.destroyAllWindows()

            # ============ SHOW CSV ============
            import tkinter
            root = tkinter.Tk()
            root.title("Attendance of " + sub)
            root.configure(background="black")

            with open(fileName, newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                r = 0
                for col in reader:
                    c = 0
                    for row in col:
                        label = tkinter.Label(
                            root, width=12, height=1, fg="yellow",
                            font=("times", 15, " bold "), bg="black",
                            text=row, relief=tkinter.RIDGE,
                        )
                        label.grid(row=r, column=c)
                        c += 1
                    r += 1

            root.mainloop()

        except Exception as ex:
            print("ERROR in FillAttendance:", ex)
            import traceback
            traceback.print_exc()
            text_to_speech(f"Error: {ex}")
            try:
                cam.release()
            except:
                pass
            cv2.destroyAllWindows()

    # ============ SUBJECT UI ============
    subject = Tk()
    subject.title("Subject...")
    subject.geometry("580x320")
    subject.resizable(0, 0)
    subject.configure(background="black")

    titl = tk.Label(subject, bg="black", relief=RIDGE, bd=10, font=("arial", 30))
    titl.pack(fill=X)

    titl = tk.Label(
        subject,
        text="Enter the Subject Name",
        bg="black",
        fg="green",
        font=("arial", 25),
    )
    titl.place(x=160, y=12)

    Notifica = tk.Label(
        subject,
        text="Attendance filled Successfully",
        bg="yellow",
        fg="black",
        width=33,
        height=2,
        font=("times", 15, "bold"),
    )

    def Attf():
        sub_name = tx.get().strip()
        if sub_name == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
        else:
            os.startfile(f"Attendance\\{sub_name}")

    attf = tk.Button(
        subject,
        text="Check Sheets",
        command=Attf,
        bd=7,
        font=("times new roman", 15),
        bg="black",
        fg="yellow",
        height=2,
        width=10,
        relief=RIDGE,
    )
    attf.place(x=360, y=170)

    sub_label = tk.Label(
        subject,
        text="Enter Subject",
        width=10,
        height=2,
        bg="black",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("times new roman", 15),
    )
    sub_label.place(x=50, y=100)

    tx = tk.Entry(
        subject,
        width=15,
        bd=5,
        bg="black",
        fg="yellow",
        relief=RIDGE,
        font=("times", 30, "bold"),
    )
    tx.place(x=190, y=100)

    fill_a = tk.Button(
        subject,
        text="Fill Attendance",
        command=FillAttendance,
        bd=7,
        font=("times new roman", 15),
        bg="black",
        fg="yellow",
        height=2,
        width=12,
        relief=RIDGE,
    )
    fill_a.place(x=195, y=170)

    subject.mainloop()
