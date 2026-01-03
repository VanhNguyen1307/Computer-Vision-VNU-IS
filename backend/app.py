from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import cv2
import numpy as np
import uuid
import threading
import time
import datetime
import pandas as pd
import shutil

# ===== YOUR PIPELINE =====
from core.yolo_face_detector import YOLOFaceDetector
from core.facenet_embedder import FaceNetEmbedder
from core.matcher import EmbeddingMatcher
from core.liveness_blink import BlinkLiveness

# import build embedding script
from scripts.build_embeddings import build_one_student


app = FastAPI()

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

studentdetail_path = "./StudentDetails/studentdetails.csv"
REGISTER_DIR = os.path.join("data", "registered_faces")

print(">>> Serving index.html from:", os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# =============================
# ATTENDANCE SESSION STORE
# =============================
SESSION_LOCK = threading.Lock()
sessions = {}  # session_id -> {subject, start_time, records, csv_path}


# =============================
# REGISTER SESSION STORE
# =============================
REGISTER_LOCK = threading.Lock()
register_sessions = {}  # register_id -> {subject,enrollment,name,folder,saved}


# ============================================================
# ✅ INIT PIPELINE ONCE
# ============================================================
detector = YOLOFaceDetector(model_path="models/yolov8n-face-lindevs.pt", conf=0.5)
embedder = FaceNetEmbedder()
matcher = EmbeddingMatcher(embeddings_dir="data/embeddings", threshold=0.55)
liveness = BlinkLiveness(timeout=10.0, ear_thresh=0.25, consec_frames=2)

current_id = None


# ============================================================
# ✅ HELPER: load student map (Enrollment -> Name)
# ============================================================
def load_student_map():
    if not os.path.exists(studentdetail_path):
        return {}
    try:
        df = pd.read_csv(studentdetail_path)
        mp = {}
        for _, row in df.iterrows():
            mp[str(row["Enrollment"]).strip()] = str(row["Name"]).strip()
        return mp
    except:
        return {}


# ============================================================
# ✅ API: process frame (attendance)
# ============================================================
@app.post("/api/frame")
async def process_frame(
    file: UploadFile = File(...),
    session_id: str = Form(None)
):
    global current_id
    student_map = load_student_map()

    try:
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"ok": False, "error": "Cannot decode image"})

        faces = detector.detect(frame)

        if len(faces) == 0:
            current_id = None
            liveness.reset()
            return {"ok": True, "found": False, "message": "No face detected"}

        faces.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
        box = faces[0]
        face_img = detector.crop_face(frame, box, margin=10)

        if face_img is None or face_img.size == 0:
            return {"ok": True, "found": False, "message": "Face crop failed"}

        emb = embedder.embed(face_img)
        sid, score = matcher.match(emb)

        if sid is None:
            current_id = None
            liveness.reset()
            return {
                "ok": True,
                "found": True,
                "recognized": False,
                "box": box,
                "message": f"Unknown ({score:.2f})"
            }

        sid_str = str(sid)
        name = student_map.get(sid_str, sid_str)

        # reset liveness if new person appears
        if current_id != sid_str:
            current_id = sid_str
            liveness.reset()

        live, info = liveness.update(face_img, need_blinks=2)

        # ✅ if live -> save record into session
        if live and session_id:
            with SESSION_LOCK:
                if session_id in sessions:
                    sessions[session_id]["records"][sid_str] = {
                        "Enrollment": sid_str,
                        "Name": name,
                        "Score": float(score),
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Liveness": True,
                    }

        return {
            "ok": True,
            "found": True,
            "recognized": True,
            "id": sid_str,
            "name": name,
            "score": float(score),
            "box": box,
            "liveness": live,
            "info": info
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)})


# ============================================================
# ✅ SESSION START/STOP
# ============================================================
@app.post("/api/session/start")
def start_session(payload: dict):
    subject = (payload.get("subject") or "").strip()
    if not subject:
        return {"ok": False, "error": "Subject is required"}

    session_id = str(uuid.uuid4())[:8]

    with SESSION_LOCK:
        sessions[session_id] = {
            "subject": subject,
            "start_time": time.time(),
            "records": {},
            "csv_path": None,
        }

    return {"ok": True, "session_id": session_id, "subject": subject}


@app.post("/api/session/stop")
def stop_session(payload: dict):
    session_id = (payload.get("session_id") or "").strip()
    if not session_id:
        return {"ok": False, "error": "session_id is required"}

    with SESSION_LOCK:
        if session_id not in sessions:
            return {"ok": False, "error": "Session not found"}

        session = sessions[session_id]
        subject = session["subject"]
        records = list(session["records"].values())

    folder = os.path.join("Attendance", subject)
    os.makedirs(folder, exist_ok=True)

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H-%M-%S")

    filename = f"{subject}_{date}_{timeStamp}.csv"
    csv_path = os.path.join(folder, filename)

    df = pd.DataFrame(records, columns=["Enrollment", "Name", "Score", "Time", "Liveness"])
    df.to_csv(csv_path, index=False)

    with SESSION_LOCK:
        sessions[session_id]["csv_path"] = csv_path

    return {"ok": True, "csv_path": csv_path, "download_url": f"/api/session/download/{session_id}"}


@app.get("/api/session/download/{session_id}")
def download_csv(session_id: str):
    with SESSION_LOCK:
        if session_id not in sessions:
            return {"ok": False, "error": "Session not found"}
        csv_path = sessions[session_id].get("csv_path")

    if not csv_path or not os.path.exists(csv_path):
        return {"ok": False, "error": "CSV not generated yet"}

    return FileResponse(csv_path, filename=os.path.basename(csv_path))


# ============================================================
# ✅ REGISTER START / CAPTURE / FINISH / CANCEL
# ============================================================
@app.post("/api/register/start")
def register_start(payload: dict):
    subject = (payload.get("subject") or "").strip()
    enrollment = (payload.get("enrollment") or "").strip()
    name = (payload.get("name") or "").strip()

    if not subject or not enrollment or not name:
        return {"ok": False, "error": "subject, enrollment, name required"}

    reg_id = str(uuid.uuid4())[:8]
    folder = os.path.join(REGISTER_DIR, subject, enrollment)
    os.makedirs(folder, exist_ok=True)

    with REGISTER_LOCK:
        register_sessions[reg_id] = {
            "subject": subject,
            "enrollment": enrollment,
            "name": name,
            "folder": folder,
            "saved": 0
        }

    # ✅ save student detail into CSV (append if new)
    os.makedirs("StudentDetails", exist_ok=True)
    if not os.path.exists(studentdetail_path):
        pd.DataFrame(columns=["Enrollment", "Name"]).to_csv(studentdetail_path, index=False)

    df = pd.read_csv(studentdetail_path)
    if enrollment not in df["Enrollment"].astype(str).values:
        df.loc[len(df)] = [enrollment, name]
        df.to_csv(studentdetail_path, index=False)

    return {"ok": True, "register_id": reg_id}


@app.post("/api/register/frame")
async def register_capture(
    file: UploadFile = File(...),
    register_id: str = Form(None)
):
    if not register_id:
        return {"ok": False, "error": "register_id is required"}

    with REGISTER_LOCK:
        if register_id not in register_sessions:
            return {"ok": False, "error": "register session not found"}
        session = register_sessions[register_id]

    try:
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"ok": False, "error": "Cannot decode image"}

        faces = detector.detect(frame)
        if len(faces) == 0:
            return {"ok": True, "saved": False, "message": "No face detected"}

        faces.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
        box = faces[0]
        face_img = detector.crop_face(frame, box, margin=10)

        if face_img is None or face_img.size == 0:
            return {"ok": True, "saved": False, "message": "Face crop failed"}

        with REGISTER_LOCK:
            session["saved"] += 1
            idx = session["saved"]
            folder = session["folder"]

        save_path = os.path.join(folder, f"{session['enrollment']}_{idx}.jpg")
        cv2.imwrite(save_path, face_img)

        return {"ok": True, "saved": True, "count": idx, "message": f"Saved {idx}"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.post("/api/register/finish")
def register_finish(payload: dict):
    reg_id = (payload.get("register_id") or "").strip()
    if not reg_id:
        return {"ok": False, "error": "register_id required"}

    with REGISTER_LOCK:
        if reg_id not in register_sessions:
            return {"ok": False, "error": "register session not found"}
        session = register_sessions.pop(reg_id)

    subject = session["subject"]
    enrollment = session["enrollment"]
    name = session["name"]

    try:
        emb_path = build_one_student(subject, enrollment)
        matcher.load_gallery()

        return {
            "ok": True,
            "enrollment": enrollment,
            "name": name,
            "embedding_path": emb_path
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/register/cancel")
def register_cancel(payload: dict):
    reg_id = (payload.get("register_id") or "").strip()
    if not reg_id:
        return {"ok": False, "error": "register_id required"}

    with REGISTER_LOCK:
        if reg_id in register_sessions:
            register_sessions.pop(reg_id)

    return {"ok": True}


# ============================================================
# ✅ LIST STUDENTS BY SUBJECT
# ============================================================
@app.get("/api/students")
def list_students(subject: str):
    subject = (subject or "").strip()
    if not subject:
        return {"ok": False, "error": "subject is required"}

    folder = os.path.join("data", "registered_faces", subject)
    if not os.path.exists(folder):
        return {"ok": True, "students": []}

    df = None
    if os.path.exists(studentdetail_path):
        df = pd.read_csv(studentdetail_path)

    students = []
    for sid in os.listdir(folder):
        sid_folder = os.path.join(folder, sid)
        if not os.path.isdir(sid_folder):
            continue

        enrollment = sid
        name = enrollment

        if df is not None:
            row = df[df["Enrollment"].astype(str) == str(enrollment)]
            if len(row) > 0:
                name = str(row.iloc[0]["Name"])

        students.append({"enrollment": enrollment, "name": name})

    return {"ok": True, "students": students}


# ============================================================
# ✅ DELETE STUDENT (subject + enrollment)
# ============================================================
@app.post("/api/student/delete")
def delete_student(payload: dict):
    subject = (payload.get("subject") or "").strip()
    enrollment = (payload.get("enrollment") or "").strip()

    if not subject or not enrollment:
        return {"ok": False, "error": "subject and enrollment required"}

    folder = os.path.join("data", "registered_faces", subject, enrollment)
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)

    emb_file = os.path.join("data", "embeddings", f"{enrollment}.pt")
    if os.path.exists(emb_file):
        os.remove(emb_file)

    matcher.load_gallery()

    return {"ok": True, "message": f"Deleted {enrollment}"}
