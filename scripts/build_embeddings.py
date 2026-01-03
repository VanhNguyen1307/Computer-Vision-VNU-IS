import os
import cv2
import torch
import glob
import numpy as np
from tqdm import tqdm

from core.yolo_face_detector import YOLOFaceDetector
from core.facenet_embedder import FaceNetEmbedder

REGISTER_DIR = os.path.join("data", "registered_faces")
EMB_DIR = os.path.join("data", "embeddings")


def build_one_student(subject: str, enrollment: str):
    """
    Build embedding for 1 student inside:
    data/registered_faces/<subject>/<enrollment>/
    """
    subject = subject.strip()
    enrollment = enrollment.strip()

    folder = os.path.join(REGISTER_DIR, subject, enrollment)
    if not os.path.exists(folder):
        raise RuntimeError(f"Student folder not found: {folder}")

    os.makedirs(EMB_DIR, exist_ok=True)

    detector = YOLOFaceDetector(model_path="models/yolov8n-face-lindevs.pt", conf=0.5)
    embedder = FaceNetEmbedder()

    embs = []

    imgs = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    if len(imgs) == 0:
        raise RuntimeError("No images found in student folder")

    for img_name in tqdm(imgs, desc=f"Embedding {subject}/{enrollment}"):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = detector.detect(img)
        if len(faces) == 0:
            continue

        faces.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
        box = faces[0]
        face_img = detector.crop_face(img, box, margin=10)

        if face_img is None or face_img.size == 0:
            continue

        emb = embedder.embed(face_img)
        embs.append(emb.unsqueeze(0))

    if len(embs) == 0:
        raise RuntimeError("No valid face embeddings could be extracted")

    embs = torch.cat(embs, dim=0)

    save_path = os.path.join(EMB_DIR, f"{enrollment}.pt")
    torch.save(embs, save_path)

    return save_path


def build_all_students():
    """
    Build embeddings for all students under data/registered_faces/*
    """
    if not os.path.exists(REGISTER_DIR):
        raise RuntimeError("Registered_faces folder not found")

    subjects = [s for s in os.listdir(REGISTER_DIR) if os.path.isdir(os.path.join(REGISTER_DIR, s))]
    if len(subjects) == 0:
        raise RuntimeError("No subjects found inside registered_faces")

    built = []

    for subject in subjects:
        subject_dir = os.path.join(REGISTER_DIR, subject)
        ids = [sid for sid in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, sid))]

        for enrollment in ids:
            try:
                p = build_one_student(subject, enrollment)
                built.append(p)
            except Exception as e:
                print(f"❌ Failed {subject}/{enrollment}: {e}")

    return built


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--enrollment", type=str, default=None)

    args = parser.parse_args()

    if args.subject and args.enrollment:
        out = build_one_student(args.subject, args.enrollment)
        print("✅ Saved embedding:", out)
    else:
        outs = build_all_students()
        print("✅ Built embeddings:", len(outs))
def build_one_student(subject: str, enrollment: str,
                      registered_dir="data/registered_faces",
                      embeddings_dir="data/embeddings"):
    """
    Build embedding .pt for ONE student after registering.
    Return path of embedding file.
    """
    subject = str(subject).strip()
    enrollment = str(enrollment).strip()

    if not subject or not enrollment:
        raise ValueError("subject and enrollment required")

    folder = os.path.join(registered_dir, subject, enrollment)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"No registered faces folder: {folder}")

    img_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No face images found in: {folder}")

    os.makedirs(embeddings_dir, exist_ok=True)

    embedder = FaceNetEmbedder()

    embs = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue

        emb = embedder.embed(img)   # torch tensor (512,)
        if emb is None:
            continue

        # normalize
        emb = emb / emb.norm()
        embs.append(emb.unsqueeze(0))

    if len(embs) == 0:
        raise RuntimeError("Could not extract embeddings from images")

    embs = torch.cat(embs, dim=0)   # (N,512)

    out_path = os.path.join(embeddings_dir, f"{enrollment}.pt")
    torch.save(embs, out_path)

    print(f"✅ Built embedding for {enrollment} ({len(img_paths)} imgs) -> {out_path}")
    return out_path