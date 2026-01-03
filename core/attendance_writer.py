import os
import csv
import time
from datetime import datetime

class AttendanceWriter:
    def __init__(self, attendance_dir="data/attendance", cooldown_sec=600):
        self.attendance_dir = attendance_dir
        os.makedirs(self.attendance_dir, exist_ok=True)

        self.cooldown_sec = cooldown_sec
        self.last_seen = {}  # sid -> last timestamp

    def mark(self, sid, name=None):
        """
        Return: (ok: bool, message: str)
        """
        now = time.time()

        if sid in self.last_seen and (now - self.last_seen[sid]) < self.cooldown_sec:
            return False, "Cooldown active"

        self.last_seen[sid] = now

        date_str = datetime.now().strftime("%Y-%m-%d")
        csv_path = os.path.join(self.attendance_dir, f"{date_str}.csv")
        file_exists = os.path.exists(csv_path)

        if name is None:
            name = f"Student_{sid}"

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ID", "Name", "Time"])
            writer.writerow([sid, name, datetime.now().strftime("%H:%M:%S")])

        return True, "Attendance marked ✅"
