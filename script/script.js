// =============================
//   GARBAGE CLASSIFICATION APP
// =============================

// --- LABELS SESUAI MODEL ---
const labels = [
  "battery", "biological", "cardboard", "clothes",
  "glass", "metal", "paper", "plastic", "shoes", "trash"
];

// --- LOAD MODEL ---
let session;

// Disable upload dulu sampai model siap
document.getElementById("fileInput").disabled = true;
document.getElementById("cameraBtn").disabled = true;

async function loadModel() {
  try {
    const modelStatus = document.getElementById("modelStatus");
    modelStatus.innerText = "⏳ Loading model...";
    modelStatus.style.color = "#888";

    // Load model ONNX
    session = await ort.InferenceSession.create("model/model_resnet18.onnx");
    console.log("✅ ONNX model loaded");
    console.log("Input names:", session.inputNames);
    console.log("Output names:", session.outputNames);

    // Update tampilan status
    modelStatus.innerText = "✅ Model loaded successfully";
    modelStatus.style.color = "#2ecc71"; // hijau lembut

    // Aktifkan upload setelah model siap
    document.getElementById("fileInput").disabled = false;
    document.getElementById("cameraBtn").disabled = false;

    // Hilangkan tulisan setelah 2.5 detik
    setTimeout(() => {
      modelStatus.innerText = "";
    }, 1500);

  } catch (err) {
    console.error("❌ Gagal load model:", err);
    const modelStatus = document.getElementById("modelStatus");
    modelStatus.innerText = "❌ Failed to load model";
    modelStatus.style.color = "#e74c3c"; // merah lembut
  }
}

loadModel();

// =============================
//   UPLOAD HANDLER
// =============================
document.getElementById('fileInput').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = document.getElementById('preview');
    const placeholder = document.getElementById('placeholder');

    placeholder.classList.add('hidden');
    preview.classList.remove('hidden');

    preview.src = e.target.result;
    preview.onload = () => runInference(preview);
  };

  reader.readAsDataURL(file);
});


// =============================
//   CAMERA HANDLER (2 tahap)
// =============================
let stream = null;
let videoElement = null;

document.getElementById('cameraBtn').addEventListener('click', async () => {
  const placeholder = document.getElementById('placeholder');
  const preview = document.getElementById('preview');
  const uploadArea = document.getElementById('uploadArea');
  const controls = document.querySelector('.controls');

  try {
    // 🔸 Jika kamera sudah aktif, jangan buka dua kali
    if (stream) return;

    // 🔹 Gunakan kamera belakang jika ada
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } }, // 👈 kamera belakang
      audio: false
    });

    // 🔥 Sembunyikan placeholder & preview sementara
    placeholder.classList.add('hidden');
    preview.classList.add('hidden');

    // 🔹 Buat elemen video (live preview)
    videoElement = document.createElement('video');
    videoElement.autoplay = true;
    videoElement.playsInline = true;
    videoElement.srcObject = stream;

    // Styling agar pas di card
    Object.assign(videoElement.style, {
      width: "100%",
      maxHeight: "320px",
      borderRadius: "8px",
      objectFit: "cover",
      marginBottom: "8px",
      boxShadow: "0 2px 6px rgba(0,0,0,0.2)"
    });

    // Tambahkan sebelum tombol-tombol
    uploadArea.insertBefore(videoElement, controls);

    // 🔹 Tombol Capture
    const captureBtn = document.createElement('button');
    captureBtn.innerText = "📸 Capture Photo";
    captureBtn.className = "btn";
    captureBtn.style.marginTop = "8px";
    controls.insertBefore(captureBtn, controls.firstChild);

    // Tunggu kamera siap
    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        videoElement.play();
        resolve();
      };
    });

    // Event Capture
    captureBtn.addEventListener('click', () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

      // 🔹 Stop semua track kamera
      stream.getTracks().forEach(track => track.stop());
      stream = null;

      // 🔹 Hapus video & tombol capture
      videoElement.remove();
      captureBtn.remove();

      // 🔹 Tampilkan hasil ke preview
      preview.src = canvas.toDataURL("image/jpeg", 0.9);
      preview.classList.remove('hidden');
      placeholder.classList.add('hidden');

      // Jalankan inferensi
      preview.onload = () => runInference(preview);
    });

  } catch (err) {
    console.error("❌ Gagal akses kamera:", err);
    alert("Tidak bisa mengakses kamera. Pastikan browser mengizinkan akses kamera.");
  }
});



// =============================
//   INFERENCE PIPELINE
// =============================
async function runInference(imgElement) {
  if (!session) {
    const container = document.getElementById("predictionsList");
    container.innerHTML = `
      <div class="pred-item">
        <div class="pred-name">⚙️ Model is still loading...</div>
        <div class="confidence"><span class="confBar" style="width:0%"></span></div>
      </div>
    `;
    return;
  }

  try {
    const start = performance.now();

    // --- Preprocess ---
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(imgElement, 0, 0, 224, 224);

    const { data } = ctx.getImageData(0, 0, 224, 224);
    const float32Data = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < 224 * 224; i++) {
      float32Data[i] = (data[i * 4] / 255 - 0.485) / 0.229;
      float32Data[i + 224 * 224] = (data[i * 4 + 1] / 255 - 0.456) / 0.224;
      float32Data[i + 2 * 224 * 224] = (data[i * 4 + 2] / 255 - 0.406) / 0.225;
    }

    const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);
    const feeds = { [session.inputNames[0]]: inputTensor };
    const results = await session.run(feeds);

    const end = performance.now();
    const outputName = session.outputNames[0];
    const output = Array.from(results[outputName].data);

    // --- Softmax ---
    const softmax = (arr) => {
      const maxVal = Math.max(...arr);
      const exps = arr.map(v => Math.exp(v - maxVal));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map(v => v / sum);
    };

    const probs = softmax(output);

    // --- Ambil 3 prediksi teratas ---
    const top3 = probs
      .map((p, i) => ({ label: labels[i], confidence: (p * 100).toFixed(2) }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    // --- Update UI ---
    const container = document.getElementById("predictionsList");
    container.innerHTML = ""; // Kosongkan tampilan lama

    top3.forEach((t, i) => {
      const item = document.createElement("div");
      item.className = "pred-item";
      item.innerHTML = `
        <div class="pred-name">
          <span>${i + 1}. ${t.label}</span>
          <span>${t.confidence}%</span>
        </div>
        <div class="confidence">
          <span class="confBar" style="width:${t.confidence}%"></span>
        </div>
      `;
      container.appendChild(item);
    });

  } catch (err) {
    console.error("❌ Error during inference:", err);
    const container = document.getElementById("predictionsList");
    container.innerHTML = `
      <div class="pred-item error">
        <div class="pred-name">❌ Error during inference</div>
        <div class="confText">Check console for details</div>
      </div>
    `;
  }
}






// =============================
//   FEEDBACK HANDLER
// =============================
function showFeedback(isCorrect) {
  const feedbackText = document.createElement("p");
  feedbackText.className = "feedback-message";
  feedbackText.innerText = isCorrect
    ? "✅ Thanks for your feedback!"
    : "❌ Thank you! We'll improve the model.";

  document.querySelector(".feedback").innerHTML = "";
  document.querySelector(".feedback").appendChild(feedbackText);
}


























/*
// --- Mock prediction (replace with real inference call) ---
async function predictImage(file){
// show loading
predClass.textContent = 'Predicting...'; confBar.style.width = '0%'; confText.textContent=''; feedbackControls.classList.add('hidden');


// simulate wait
await new Promise(r=>setTimeout(r,700));


// simulate prediction (random) — replace this block to call your model endpoint
const predicted = CLASSES[Math.floor(Math.random()*CLASSES.length)];
const confidence = Math.floor(60 + Math.random()*38); // 60-98


// render
predClass.textContent = predicted;
confBar.style.width = confidence + '%';
confText.textContent = confidence + '% confidence';
feedbackControls.classList.remove('hidden');


// save a prediction entry for stats
saveEntry({type:'pred', predicted, confidence, ts:new Date().toISOString()});


// preselect correction dropdown default to first non-equal class
correctSelect.selectedIndex = 0;


// attach quick feedback handlers
btnCorrect.onclick = ()=>{
saveEntry({type:'feedback', pred:predicted, predicted:predicted, correct:true, ts:new Date().toISOString()});
alert('Terima kasih, feedback dicatat: model dinyatakan tepat.');
};
btnWrong.onclick = ()=>{
// open correction selector visually
correctSelect.focus();
};


submitCorrection.onclick = ()=>{
const correct = correctSelect.value;
if(!correct){ alert('Pilih kategori yang benar terlebih dahulu.'); return; }
saveEntry({type:'feedback', pred:predicted, predicted:predicted, correct:false, correctLabel:correct, ts:new Date().toISOString()});
alert('Terima kasih, koreksi telah dikirim.');
};
}


// helper: reset (not shown) but could be added
*/