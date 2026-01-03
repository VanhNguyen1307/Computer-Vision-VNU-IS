from ultralytics import YOLO

class YOLOFaceDetector:
    def __init__(self, model_path="models/yolov8n-face-lindevs.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Returns: list of dict {x1,y1,x2,y2,conf}
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        faces = []

        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                faces.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})

        return faces

    @staticmethod
    def crop_face(frame, box, margin=10):
        h, w = frame.shape[:2]
        x1 = max(0, box["x1"] - margin)
        y1 = max(0, box["y1"] - margin)
        x2 = min(w, box["x2"] + margin)
        y2 = min(h, box["y2"] + margin)
        face = frame[y1:y2, x1:x2]
        return face if face.size != 0 else None
