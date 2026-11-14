const TARGET_RATE = 22050;   // matches the model's preprocessing
const MAX_SECONDS = 30;      // longer clips are trimmed before upload
const MAX_RECORD = 15;
const MIN_RECORD = 2;

const $ = (id) => document.getElementById(id);
const els = {
  headline: $("headline"), subline: $("subline"), tabs: $("tabs"),
  upload: $("panel-upload"), record: $("panel-record"),
  drop: $("dropzone"), file: $("file"),
  rec: $("rec"), ring: $("ring"), recTime: $("rec-time"),
  busy: $("busy"), result: $("result"), error: $("error"),
  title: $("title"), confidence: $("confidence"), others: $("others"), again: $("again"),
};

// ---------- views ----------

function show(view) {
  const idle = view === "idle";
  els.tabs.hidden = !idle;
  els.upload.hidden = !idle || currentMode !== "upload";
  els.record.hidden = !idle || currentMode !== "record";
  els.busy.hidden = view !== "busy";
  els.result.hidden = view !== "result";
  els.headline.hidden = view !== "idle";
  els.subline.hidden = view !== "idle";
  if (view !== "idle") els.error.hidden = true;
}

function showError(message) {
  show("idle");
  els.error.textContent = message;
  els.error.hidden = false;
}

let currentMode = "upload";

els.tabs.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (!btn) return;
  currentMode = btn.dataset.mode;
  for (const tab of els.tabs.children) {
    tab.setAttribute("aria-selected", String(tab === btn));
  }
  els.error.hidden = true;
  show("idle");
});

els.again.addEventListener("click", () => show("idle"));

// ---------- audio → 22.05 kHz mono WAV ----------

async function toWav(blob) {
  const raw = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(raw);
  } catch {
    throw new Error("Couldn't read that audio. Try a WAV, MP3 or M4A file.");
  } finally {
    ctx.close();
  }

  const seconds = Math.min(decoded.duration, MAX_SECONDS);
  const offline = new OfflineAudioContext(1, Math.ceil(seconds * TARGET_RATE), TARGET_RATE);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return encodeWav(rendered.getChannelData(0), TARGET_RATE);
}

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

// ---------- identify ----------

async function identify(blob) {
  show("busy");
  try {
    const wav = await toWav(blob);
    const form = new FormData();
    form.append("file", wav, "clip.wav");
    const res = await fetch("/api/identify", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Something went wrong. Please try again.");
    render(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
  }
}

const pct = (x) => `${Math.round(x * 100)}%`;

function render({ matches }) {
  const [best, ...rest] = matches;
  const unsure = best.confidence < 0.5;
  els.title.textContent = best.title;
  els.confidence.textContent = unsure
    ? `Not sure — ${pct(best.confidence)} match. Try a longer or cleaner clip.`
    : `${pct(best.confidence)} match`;

  els.others.replaceChildren(
    ...rest.filter((m) => m.confidence >= 0.01).map((m) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      const score = document.createElement("span");
      const meter = document.createElement("div");
      const fill = document.createElement("b");
      name.textContent = m.title;
      score.textContent = pct(m.confidence);
      meter.className = "meter";
      fill.style.width = pct(m.confidence);
      meter.append(fill);
      li.append(name, score, meter);
      return li;
    })
  );
  show("result");
}

// ---------- upload ----------

els.file.addEventListener("change", () => {
  const f = els.file.files[0];
  els.file.value = "";
  if (f) identify(f);
});

els.drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.file.click(); }
});

["dragenter", "dragover"].forEach((t) =>
  els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add("over"); })
);
["dragleave", "drop"].forEach((t) =>
  els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.remove("over"); })
);
els.drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (!f) return;
  if (!f.type.startsWith("audio/") && !/\.(wav|mp3|m4a|flac|ogg|aac|webm)$/i.test(f.name)) {
    showError("That doesn't look like an audio file.");
    return;
  }
  identify(f);
});

// ---------- record ----------

let recorder = null;
let stream = null;
let chunks = [];
let timer = null;
let started = 0;
let audioCtx = null;
let meterFrame = 0;

function fmt(s) {
  return `0:${String(Math.floor(s)).padStart(2, "0")}`;
}

async function startRecording() {
  els.error.hidden = true;
  try {
    // Speech-oriented processing would mangle music, so switch it off.
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch {
    showError("Microphone access was blocked. Allow it in your browser and try again.");
    return;
  }

  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = onRecordingStopped;
  recorder.start();

  started = performance.now();
  els.rec.classList.add("on");
  els.rec.setAttribute("aria-label", "Stop recording");
  timer = setInterval(() => {
    const elapsed = (performance.now() - started) / 1000;
    els.recTime.textContent = `${fmt(elapsed)} / ${fmt(MAX_RECORD)}`;
    if (elapsed >= MAX_RECORD) stopRecording();
  }, 100);
  els.recTime.textContent = `0:00 / ${fmt(MAX_RECORD)}`;

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  const data = new Uint8Array(analyser.fftSize);
  const tick = () => {
    analyser.getByteTimeDomainData(data);
    let peak = 0;
    for (const b of data) peak = Math.max(peak, Math.abs(b - 128));
    els.ring.style.transform = `scale(${1 + Math.min(peak / 128, 1) * 0.35})`;
    meterFrame = requestAnimationFrame(tick);
  };
  tick();
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

function cleanupRecording() {
  clearInterval(timer);
  cancelAnimationFrame(meterFrame);
  stream?.getTracks().forEach((t) => t.stop());
  audioCtx?.close();
  stream = audioCtx = null;
  els.ring.style.transform = "";
  els.rec.classList.remove("on");
  els.rec.setAttribute("aria-label", "Start recording");
  els.recTime.textContent = "Tap to record";
}

function onRecordingStopped() {
  const seconds = (performance.now() - started) / 1000;
  const blob = new Blob(chunks, { type: recorder.mimeType });
  cleanupRecording();
  if (seconds < MIN_RECORD) {
    showError("That was too short. Record at least a few seconds.");
    return;
  }
  identify(blob);
}

els.rec.addEventListener("click", () => {
  if (recorder && recorder.state === "recording") stopRecording();
  else startRecording();
});

show("idle");
