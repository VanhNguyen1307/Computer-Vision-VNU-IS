// ===============================
// GLOBAL CAMERA + CANVAS
// ===============================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Attendance UI
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");

// Register UI
const regEnrollment = document.getElementById("regEnrollment");
const regName = document.getElementById("regName");
const regSubject = document.getElementById("regSubject");
const regStartBtn = document.getElementById("regStartBtn");
const regStopBtn = document.getElementById("regStopBtn");
const regStatusText = document.getElementById("regStatusText");

// Delete UI
const delSubject = document.getElementById("delSubject");
const loadStudentBtn = document.getElementById("loadStudentBtn");
const studentSelect = document.getElementById("studentSelect");
const deleteStudentBtn = document.getElementById("deleteStudentBtn");
const deleteStatusText = document.getElementById("deleteStatusText");

// ===============================
// CAMERA STATE
// ===============================
let stream = null;

async function startCamera() {
  if (stream) return;

  stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false
  });

  video.srcObject = stream;

  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  };
}

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach(t => t.stop());
  stream = null;
}

// draw bbox
function drawBox(box, text, color = "lime") {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);

  ctx.fillStyle = color;
  ctx.font = "20px Arial";
  ctx.fillText(text, box.x1, Math.max(20, box.y1 - 10));
}

// ===============================
// ATTENDANCE MODE
// ===============================
let running = false;
let timer = null;
let sessionId = null;

// send attendance frame
async function sendFrame() {
  if (!running) return;
  if (video.readyState < 2) return;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise(resolve =>
    canvas.toBlob(resolve, "image/jpeg", 0.8)
  );

  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");
  if (sessionId) formData.append("session_id", sessionId);

  try {
    const res = await fetch("/api/frame", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (!data.ok) {
      statusText.textContent = `❌ Error: ${data.error}`;
      return;
    }

    if (!data.found) {
      statusText.textContent = data.message || "No face detected";
      return;
    }

    if (!data.recognized) {
      drawBox(data.box, data.message, "red");
      statusText.textContent = data.message;
      return;
    }

    const txt = `${data.id} (${data.score.toFixed(2)})`;
    drawBox(data.box, txt, data.liveness ? "lime" : "yellow");

    statusText.textContent =
      `ID: ${data.id}\nName: ${data.name || ""}\nScore: ${data.score.toFixed(
        3
      )}\nLiveness: ${data.liveness}\n${data.info}`;
  } catch (err) {
    statusText.textContent = `❌ Fetch error: ${err}`;
  }
}

// ✅ START ATTENDANCE
startBtn.addEventListener("click", async () => {
  const subject = document.getElementById("subject").value.trim();
  if (!subject) {
    alert("Please enter subject!");
    return;
  }

  try {
    const res = await fetch("/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject })
    });

    const data = await res.json();
    if (!data.ok) {
      alert("❌ Cannot start session: " + data.error);
      return;
    }

    sessionId = data.session_id;
  } catch (err) {
    alert("❌ Start session error: " + err);
    return;
  }

  await startCamera();

  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;

  statusText.textContent =
    `✅ Attendance started for subject: ${subject}\nSession: ${sessionId}\nSending frames...`;

  timer = setInterval(sendFrame, 250);
});

// ✅ STOP ATTENDANCE
stopBtn.addEventListener("click", async () => {
  running = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;

  if (timer) clearInterval(timer);
  timer = null;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  statusText.textContent = "Stopping... saving attendance...";

  if (sessionId) {
    try {
      const stopRes = await fetch("/api/session/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });

      const stopData = await stopRes.json();

      if (stopData.ok) {
        statusText.textContent =
          `✅ Attendance saved!\nCSV: ${stopData.csv_path}\nDownload: http://localhost:8000${stopData.download_url}`;
      } else {
        statusText.textContent = `❌ Stop error: ${stopData.error}`;
      }
    } catch (err) {
      statusText.textContent = `❌ Stop fetch error: ${err}`;
    }
  } else {
    statusText.textContent = "Stopped. (No sessionId)";
  }

  sessionId = null;
  stopCamera();
});

// ===============================
// REGISTER MODE (AUTO CAPTURE)
// ===============================
let registerId = null;
let regRunning = false;
let regTimer = null;
let regCount = 0;
const REG_TARGET = 10;

// ✅ send register frame (IMPORTANT: must call /api/register/capture)
async function sendRegisterFrame() {
  if (!regRunning) return;
  if (!registerId) return;
  if (video.readyState < 2) return;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise(resolve =>
    canvas.toBlob(resolve, "image/jpeg", 0.85)
  );

  const formData = new FormData();
  formData.append("file", blob, "reg.jpg");
  formData.append("register_id", registerId);

  try {
    const res = await fetch("/api/register/capture", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!data.ok) {
      regStatusText.textContent = `❌ Capture error: ${data.error}`;
      return;
    }

    if (!data.saved) {
      regStatusText.textContent =
        `⚠️ ${data.message || "No face detected"}\nSaved ${regCount}/${REG_TARGET}`;
      return;
    }

    regCount = data.count;
    regStatusText.textContent = `✅ Saved ${regCount}/${REG_TARGET}`;

    if (regCount >= REG_TARGET) {
      regRunning = false;
      clearInterval(regTimer);
      regTimer = null;

      regStatusText.textContent =
        `✅ Saved ${regCount}/${REG_TARGET}\n⏳ Building embedding...`;

      await finishRegister();
    }
  } catch (err) {
    regStatusText.textContent = `❌ Capture fetch error: ${err}`;
  }
}

// ✅ finish register
async function finishRegister() {
  if (!registerId) return;

  try {
    const res = await fetch("/api/register/finish", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ register_id: registerId })
    });

    const data = await res.json();

    if (!data.ok) {
      regStatusText.textContent = `❌ Finish error: ${data.error}`;
      regStartBtn.disabled = false;
      return;
    }

    regStatusText.textContent =
      `🎉 Register completed!\nStudent: ${data.enrollment}\nEmbedding built ✅`;
  } catch (err) {
    regStatusText.textContent = `❌ Finish fetch error: ${err}`;
  }

  registerId = null;
  regCount = 0;
  regRunning = false;

  regStartBtn.disabled = false;
  regStopBtn.disabled = true;
  stopCamera();
}

// ✅ start register
regStartBtn.addEventListener("click", async () => {
  const enrollment = regEnrollment.value.trim();
  const name = regName.value.trim();
  const subject = regSubject.value.trim();

  if (!subject || !enrollment || !name) {
    alert("Please enter Subject + Enrollment + Name!");
    return;
  }

  regStatusText.textContent = "⏳ Starting register session...";
  regCount = 0;

  try {
    const res = await fetch("/api/register/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enrollment, name, subject })
    });

    const data = await res.json();

    if (!data.ok) {
      regStatusText.textContent = `❌ Start error: ${data.error}`;
      return;
    }

    registerId = data.register_id;
    regStatusText.textContent =
      `✅ Register started!\nID: ${registerId}\nAuto capturing faces...`;

  } catch (err) {
    regStatusText.textContent = `❌ Start fetch error: ${err}`;
    return;
  }

  await startCamera();

  regRunning = true;
  regStartBtn.disabled = true;
  regStopBtn.disabled = false;

  regTimer = setInterval(sendRegisterFrame, 700);
});

// ✅ stop register
regStopBtn.addEventListener("click", async () => {
  regRunning = false;

  if (regTimer) clearInterval(regTimer);
  regTimer = null;

  regStatusText.textContent = "⛔ Register stopped (not completed).";
  regStopBtn.disabled = true;
  regStartBtn.disabled = false;

  if (registerId) {
    try {
      await fetch("/api/register/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ register_id: registerId })
      });
    } catch (err) {}
  }

  registerId = null;
  regCount = 0;
  stopCamera();
});

// ===============================
// DELETE MODE
// ===============================
loadStudentBtn.addEventListener("click", async () => {
  const subject = delSubject.value.trim();
  if (!subject) {
    alert("Please enter subject!");
    return;
  }

  deleteStatusText.textContent = "⏳ Loading students...";

  try {
    const res = await fetch(`/api/students?subject=${encodeURIComponent(subject)}`);
    const data = await res.json();

    if (!data.ok) {
      deleteStatusText.textContent = `❌ Error: ${data.error}`;
      return;
    }

    studentSelect.innerHTML = `<option value="">-- Select student --</option>`;
    data.students.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s.enrollment;
      opt.textContent = `${s.enrollment} - ${s.name}`;
      studentSelect.appendChild(opt);
    });

    deleteStatusText.textContent = `✅ Loaded ${data.students.length} students`;

  } catch (err) {
    deleteStatusText.textContent = `❌ Fetch error: ${err}`;
  }
});

deleteStudentBtn.addEventListener("click", async () => {
  const subject = delSubject.value.trim();
  const enrollment = studentSelect.value;

  if (!subject || !enrollment) {
    alert("Please choose student!");
    return;
  }

  if (!confirm(`Delete student ${enrollment}?`)) return;

  deleteStatusText.textContent = "⏳ Deleting...";

  try {
    const res = await fetch("/api/student/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, enrollment })
    });

    const data = await res.json();

    if (!data.ok) {
      deleteStatusText.textContent = `❌ Error: ${data.error}`;
      return;
    }

    deleteStatusText.textContent = `✅ Deleted ${enrollment}`;
    loadStudentBtn.click();

  } catch (err) {
    deleteStatusText.textContent = `❌ Fetch error: ${err}`;
  }
});
// ============================
// ✅ Student List + Delete UI
// ============================

const loadStudentsBtn = document.getElementById("loadStudentsBtn");
const listSubjectInput = document.getElementById("listSubject");
const studentsBox = document.getElementById("studentsBox");

async function loadStudents() {
  const subject = listSubjectInput.value.trim();
  if (!subject) {
    alert("Please enter subject to load students!");
    return;
  }

  studentsBox.innerHTML = "⏳ Loading students...";

  try {
    const res = await fetch(`/api/students?subject=${encodeURIComponent(subject)}`);
    const data = await res.json();

    if (!data.ok) {
      studentsBox.innerHTML = `❌ Error: ${data.error}`;
      return;
    }

    if (!data.students || data.students.length === 0) {
      studentsBox.innerHTML = "✅ No students registered for this subject.";
      return;
    }

    // render list
    let html = `<table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%; color: white;">
        <tr style="background:#333;">
          <th>Enrollment</th>
          <th>Name</th>
          <th>Action</th>
        </tr>`;

    data.students.forEach(st => {
      html += `
        <tr>
          <td>${st.enrollment}</td>
          <td>${st.name}</td>
          <td>
            <button onclick="deleteStudent('${subject}', '${st.enrollment}')"
              style="background:red; color:white; padding:6px 12px; border:none; cursor:pointer;">
              Delete
            </button>
          </td>
        </tr>`;
    });

    html += `</table>`;
    studentsBox.innerHTML = html;

  } catch (err) {
    studentsBox.innerHTML = `❌ Fetch error: ${err}`;
  }
}

async function deleteStudent(subject, enrollment) {
  if (!confirm(`Delete student ${enrollment} in subject "${subject}"?`)) return;

  studentsBox.innerHTML = `⏳ Deleting ${enrollment}...`;

  try {
    const res = await fetch("/api/student/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, enrollment })
    });

    const data = await res.json();

    if (!data.ok) {
      studentsBox.innerHTML = `❌ Delete error: ${data.error}`;
      return;
    }

    studentsBox.innerHTML = `✅ Deleted ${enrollment}. Reloading list...`;
    setTimeout(loadStudents, 800);

  } catch (err) {
    studentsBox.innerHTML = `❌ Delete fetch error: ${err}`;
  }
}

if (loadStudentsBtn) {
  loadStudentsBtn.addEventListener("click", loadStudents);
}

// ✅ Auto-load students when typing subject (debounce)
let loadTimer = null;
if (listSubjectInput) {
  listSubjectInput.addEventListener("input", () => {
    if (loadTimer) clearTimeout(loadTimer);

    loadTimer = setTimeout(() => {
      if (listSubjectInput.value.trim()) loadStudents();
    }, 500);
  });
}

