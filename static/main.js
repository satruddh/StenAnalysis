


const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const uploadControls = document.getElementById("uploadControls");
const changeImageBtn = document.getElementById("changeImage");
const removeImageBtn = document.getElementById("removeImage");

let activeTimestamp = null;
let activePatientId = null;
let inputImg = null;
let mask1 = null;
let mask2 = null;
let pointAnnotations = [];
let polygonAnnotations = [];

const formContainer = document.getElementById("formContainer");
const nextBtn = document.getElementById("nextBtn");
const patientIdInput = document.getElementById("patientId");
const patientNameInput = document.getElementById("patientName");
const patientAgeInput = document.getElementById("patientAge");
const patientWeightInput = document.getElementById("patientWeight");
const patientSexInput = document.getElementById("patientSex");
const patientBGInput = document.getElementById("patientBG");
const doctorIdInput = document.getElementById("doctorId");
nextBtn.addEventListener("click", () => {
  if (patientIdInput.value && patientNameInput.value && patientAgeInput.value && patientWeightInput.value && patientSexInput.value && patientBGInput.value && doctorIdInput.value) {
    formContainer.classList.add("show-upload");
  } else {
    alert("Please fill all patient details");
  }
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  let inputType = document.querySelector('input[name="inputType"]:checked').value; 
  if (file) {
    if (inputType === "dicom") {
      
      document.getElementById("imagePreview").src = "/static/file_preview.png";
    }
    else {
      imagePreview.src = URL.createObjectURL(file);
    }
    uploadControls.classList.remove("d-none");
  }
});

changeImageBtn.addEventListener("click", () => imageInput.click());

removeImageBtn.addEventListener("click", () => {
  imageInput.value = "";
  imagePreview.src = "/static/placeholder.png";
  uploadControls.classList.add("d-none");
});

function loadImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });
}

function hexToRgb(hex) {
  const bigint = parseInt(hex.slice(1), 16);
  return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
}

function drawCanvas(canvasId, baseImg, maskImg, opacity = 1.0, mode = "mask", smooth = false, overlayColor = "#ff0000") {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  
  if (mode === "mask") {
    ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height);

    if (opacity > 0) {
      const tmpCanvas = document.createElement("canvas");
      tmpCanvas.width = canvas.width;
      tmpCanvas.height = canvas.height;
      const tmpCtx = tmpCanvas.getContext("2d");

      tmpCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
      const maskData = tmpCtx.getImageData(0, 0, canvas.width, canvas.height);
      const baseData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      const [rC, gC, bC] = hexToRgb(overlayColor);

      for (let i = 0; i < maskData.data.length; i += 4) {
        const value = maskData.data[i];
        if (value > 127) {
          baseData.data[i] = rC;
          baseData.data[i + 1] = gC;
          baseData.data[i + 2] = bC;
          baseData.data[i + 3] = Math.floor(opacity * 255);
        }
      }

      ctx.putImageData(baseData, 0, 0);
    }
  }

  
  polygonAnnotations.forEach(poly => {
    if (poly.stage === canvasId) {
      ctx.strokeStyle = poly.color || "#00ffff";
      ctx.fillStyle = poly.color ? poly.color + "55" : "rgba(0,255,255,0.3)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(poly.points[0].x, poly.points[0].y);
      poly.points.slice(1).forEach(pt => ctx.lineTo(pt.x, pt.y));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      
      const center = poly.points.reduce((acc, pt) => {
        acc.x += pt.x;
        acc.y += pt.y;
        return acc;
      }, { x: 0, y: 0 });
      center.x /= poly.points.length;
      center.y /= poly.points.length;

      const text = poly.label || "Note";
      const fontSize = 13;
      const padding = 8;
      ctx.font = `${fontSize}px Arial`;
      ctx.textBaseline = "top";
      const textWidth = ctx.measureText(text).width;
      const boxWidth = textWidth + padding * 2;
      const boxHeight = fontSize + padding * 1.5;
      const boxX = center.x;
      const boxY = center.y - boxHeight - 20;

      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const radius = 8;
      ctx.moveTo(boxX - boxWidth / 2 + radius, boxY);
      ctx.lineTo(boxX + boxWidth / 2 - radius, boxY);
      ctx.quadraticCurveTo(boxX + boxWidth / 2, boxY, boxX + boxWidth / 2, boxY + radius);
      ctx.lineTo(boxX + boxWidth / 2, boxY + boxHeight - radius);
      ctx.quadraticCurveTo(boxX + boxWidth / 2, boxY + boxHeight, boxX + boxWidth / 2 - radius, boxY + boxHeight);
      ctx.lineTo(boxX + 10, boxY + boxHeight);
      ctx.lineTo(boxX, boxY + boxHeight + 10);
      ctx.lineTo(boxX - 10, boxY + boxHeight);
      ctx.lineTo(boxX - boxWidth / 2 + radius, boxY + boxHeight);
      ctx.quadraticCurveTo(boxX - boxWidth / 2, boxY + boxHeight, boxX - boxWidth / 2, boxY + boxHeight - radius);
      ctx.lineTo(boxX - boxWidth / 2, boxY + radius);
      ctx.quadraticCurveTo(boxX - boxWidth / 2, boxY, boxX - boxWidth / 2 + radius, boxY);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.fillText(text, boxX, boxY + padding * 0.75);
    }
  });

  
  pointAnnotations.forEach(pt => {
    if (pt.stage === canvasId) {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = "yellow";
      ctx.fill();

      
      const label = pt.label || "Note";
      ctx.font = "13px Arial";
      const padding = 6;
      const textWidth = ctx.measureText(label).width;
      const boxWidth = textWidth + padding * 2;
      const boxHeight = 20;
      const x = pt.x;
      const y = pt.y - 30;

      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x - boxWidth / 2, y);
      ctx.lineTo(x + boxWidth / 2, y);
      ctx.lineTo(x + boxWidth / 2, y + boxHeight);
      ctx.lineTo(x + 10, y + boxHeight);
      ctx.lineTo(x, y + boxHeight + 10);
      ctx.lineTo(x - 10, y + boxHeight);
      ctx.lineTo(x - boxWidth / 2, y + boxHeight);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.fillText(label, x, y + 5);
    }
  });
}

function setupControls(canvasId, baseImg, maskImg, viewName, opacityId, smoothId, colorId) {
  const radios = document.querySelectorAll(`input[name="${viewName}"]`);
  const slider = document.getElementById(opacityId);
  const color = document.getElementById(colorId);
  const smooth = document.getElementById(smoothId);

  function redraw() {
    const mode = document.querySelector(`input[name="${viewName}"]:checked`).value;
    const op = parseFloat(slider.value);
    const col = color.value;
    const isSmooth = smooth.checked;
    drawCanvas(canvasId, baseImg, maskImg, op, mode, isSmooth, col);
  }

  radios.forEach(r => r.addEventListener("change", redraw));
  slider.addEventListener("input", redraw);
  color.addEventListener("input", redraw);
  smooth.addEventListener("change", redraw);
}

document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const patientId = document.getElementById("patientId").value;
  const doctorId = document.getElementById("doctorId").value || "UnknownDoctor";
  const file = imageInput.files[0];
  if (!file || !patientId) return;

  const patientName = patientNameInput.value;
  const patientAge = patientAgeInput.value;
  const patientWeight = patientWeightInput.value;
  const patientSex = patientSexInput.value;
  const patientBG = patientBGInput.value;

  const formData = new FormData();
  formData.append("image", file);
  formData.append("patient_name", patientName);
  formData.append("patient_age", patientAge);
  formData.append("patient_weight", patientWeight);
  formData.append("patient_sex", patientSex);
  formData.append("patient_bg", patientBG);

  formData.append("patient_id", patientId);
  formData.append("doctor_id", doctorId);
  const inputType = document.querySelector('input[name="inputType"]:checked').value;
  formData.append("input_type", inputType);


  const response = await fetch("/api/inference", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  if (data.error) {
    alert("Error: " + data.error);
    return;
  }

  activePatientId = patientId;
  activeTimestamp = data.timestamp;
  loadSidebarCases(patientId);

  document.getElementById("results").classList.remove("d-none");
  document.getElementById("controlsPanel").classList.remove("d-none");
  document.getElementById("legendPanel").classList.remove("d-none");
  document.getElementById("doctorNotesPanel").classList.remove("d-none");
  document.getElementById("exportBtn").classList.remove("d-none");
  document.getElementById("saveBtn").classList.remove("d-none");
  document.getElementById("patientInfo").classList.remove("d-none");
  document.getElementById("uploadSection").classList.add("d-none");
  document.getElementById("displayPatientId").textContent = patientId;
  document.getElementById("displayDoctorId").textContent = doctorId;

  inputImg = await loadImage("data:image/png;base64," + data.input_base64);
  mask1 = await loadImage("data:image/png;base64," + data.output1_base64);
  mask2 = await loadImage("data:image/png;base64," + data.output2_base64);

  const opacity1 = parseFloat(document.getElementById("opacity1").value);
  const opacity2 = parseFloat(document.getElementById("opacity2").value);
  const overlayColor1 = document.getElementById("overlayColor1").value;
  const overlayColor2 = document.getElementById("overlayColor2").value;

  const smooth1 = document.getElementById("smooth1").checked;
  const smooth2 = document.getElementById("smooth2").checked;

  drawCanvas("canvas1", inputImg, mask1, opacity1, "overlay", smooth1, overlayColor1);
  drawCanvas("canvas2", inputImg, mask2, opacity2, "overlay", smooth2, overlayColor2);

  drawCanvas("canvas1", inputImg, mask1, opacity1, "overlay", false, overlayColor1);
  drawCanvas("canvas2", inputImg, mask2, opacity2, "overlay", false, overlayColor2);

  setupControls("canvas1", inputImg, mask1, "view1", "opacity1", "smooth1", "overlayColor1");
  setupControls("canvas2", inputImg, mask2, "view2", "opacity2", "smooth2", "overlayColor2");

  setupLegendColorSync();


  document.getElementById("canvas1").addEventListener("click", () => setupFabricAnnotation("canvas1"));
  document.getElementById("canvas2").addEventListener("click", () => setupFabricAnnotation("canvas2"));
});


function createAnnotationModal() {
  const modal = document.getElementById("annotationModal");
  modal.innerHTML = "";

  const wrapper = document.createElement("div");
  wrapper.className = "annotation-content";
  wrapper.style.display = "flex";
  wrapper.style.maxHeight = "80vh";
  wrapper.style.overflow = "hidden";
  wrapper.style.gap = "20px";

  const canvasContainer = document.createElement("div");
  canvasContainer.style.flex = "3";
  canvasContainer.style.overflow = "auto";

  const canvas = document.createElement("canvas");
  canvas.id = "fabricCanvas";
  canvas.width = 512;
  canvas.height = 512;
  canvas.style.maxWidth = "100%";
  canvas.style.height = "auto";
  canvasContainer.appendChild(canvas);

  const sidebar = document.createElement("div");
  sidebar.className = "annotation-sidebar";
  sidebar.style.flex = "1";
  sidebar.innerHTML = `
    <button class="btn-close-red" id="closeAnnotationBtn">×</button>
    <label>Polygon Color:</label>
    <input type="color" id="polygonColor" value="#00ffff" class="form-control mb-2" />
    <label>Label:</label>
    <input type="text" id="polygonLabel" class="form-control mb-2" placeholder="Enter label" />
    <button id="savePolygon" class="btn btn-success">Save Annotation</button>
  `;

  wrapper.appendChild(canvasContainer);
  wrapper.appendChild(sidebar);
  modal.appendChild(wrapper);
}



function setupFabricAnnotation(canvasId) {
  const sourceCanvas = document.getElementById(canvasId);
  const modal = document.getElementById("annotationModal");

  createAnnotationModal();
  modal.classList.remove("d-none");

  const fabricCanvas = new fabric.Canvas("fabricCanvas");
  const imageData = sourceCanvas.toDataURL();

  fabric.Image.fromURL(imageData, function (img) {
    img.scaleToWidth(fabricCanvas.getWidth());
    fabricCanvas.setBackgroundImage(img, fabricCanvas.renderAll.bind(fabricCanvas));
  });

  fabricCanvas.on("mouse:down", function (opt) {
    const pointer = fabricCanvas.getPointer(opt.e);
    if (!fabricCanvas.currentPolygon) {
      const poly = new fabric.Polygon([pointer], {
        fill: "rgba(0,255,255,0.3)",
        stroke: "#00ffff",
        strokeWidth: 2,
        objectCaching: false
      });
      fabricCanvas.add(poly);
      fabricCanvas.currentPolygon = poly;
    } else {
      fabricCanvas.currentPolygon.points.push(pointer);
      fabricCanvas.currentPolygon.set({ points: fabricCanvas.currentPolygon.points });
      fabricCanvas.requestRenderAll();
    }
  });

  document.getElementById("closeAnnotationBtn").onclick = () => {
    modal.classList.add("d-none");
    fabricCanvas.dispose();
  };

  document.getElementById("savePolygon").onclick = () => {
    const label = document.getElementById("polygonLabel").value || "Unnamed";
    const color = document.getElementById("polygonColor").value || "#00ffff";
    const points = fabricCanvas.currentPolygon?.points || [];

    if (points.length > 2) {
      polygonAnnotations.push({
        label,
        color,
        stage: canvasId,
        points: points.map(p => ({ x: p.x, y: p.y }))
      });

      const opacity = parseFloat(document.getElementById(canvasId === "canvas1" ? "opacity1" : "opacity2").value);
      const smooth = document.getElementById(canvasId === "canvas1" ? "smooth1" : "smooth2").checked;
      const overlayColor = document.getElementById(canvasId === "canvas1" ? "overlayColor1" : "overlayColor2").value;
      const maskImg = canvasId === "canvas1" ? mask1 : mask2;

      drawCanvas(canvasId, inputImg, maskImg, opacity, "overlay", smooth, overlayColor);
    }

    modal.classList.add("d-none");
    fabricCanvas.dispose();
  };
}


function setupLegendColorSync() {
  const legend1 = document.getElementById("stage1legend");
  const legend2 = document.getElementById("stage2legend");
  const colorPicker1 = document.getElementById("overlayColor1");
  const colorPicker2 = document.getElementById("overlayColor2");

  function updateLegendColors() {
    legend1.style.color = colorPicker1.value;
    legend2.style.color = colorPicker2.value;
  }

  updateLegendColors();

  colorPicker1.addEventListener("input", () => {
    updateLegendColors();
    document.querySelector('input[name="view1"]:checked').dispatchEvent(new Event('change'));
  });

  colorPicker2.addEventListener("input", () => {
    updateLegendColors();
    document.querySelector('input[name="view2"]:checked').dispatchEvent(new Event('change'));
  });
}


setupLegendColorSync();

document.getElementById("exportBtn").addEventListener("click", async () => {
  const generalNotes = document.getElementById("generalNotes").value || "";
  const doctorId = document.getElementById("doctorId")?.value || "UnknownDoctor";

  
  const canvas1DataUrl = document.getElementById("canvas1").toDataURL("image/png");
  const canvas2DataUrl = document.getElementById("canvas2").toDataURL("image/png");

  
  const payload = {
    doctor_id: doctorId,
    patient_id: activePatientId,
    timestamp: activeTimestamp,
    notes: generalNotes,
    annotations: polygonAnnotations,
    images: {
      canvas1_annotated: canvas1DataUrl,
      canvas2_annotated: canvas2DataUrl
    }
  };

  try {
    const response = await fetch("/api/export", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error("Export failed.");
    }

    const blob = await response.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `export_${activePatientId}_${doctorId}.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } catch (err) {
    alert("Export failed: " + err.message);
  }
});


document.getElementById("saveBtn").addEventListener("click", async () => {
  const generalNotes = document.getElementById("generalNotes").value || "";
  const doctorId = document.getElementById("doctorId")?.value || "UnknownDoctor";

  const canvas1DataUrl = document.getElementById("canvas1").toDataURL("image/png");
  const canvas2DataUrl = document.getElementById("canvas2").toDataURL("image/png");

  const payload = {
    doctor_id: doctorId,
    patient_id: activePatientId,
    timestamp: activeTimestamp,
    notes: generalNotes,
    annotations: polygonAnnotations,
    images: {
      canvas1_annotated: canvas1DataUrl,
      canvas2_annotated: canvas2DataUrl
    }
  };

  try {
    const response = await fetch("/api/save", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const result = await response.json();
    if (response.ok) {
      alert("Analysis saved successfully!");
      window.location.reload();

    } else {
      throw new Error(result.error || "Save failed.");
    }
  } catch (err) {
    alert("Save failed: " + err.message);
  }
});


window.addEventListener("DOMContentLoaded", loadSidebarCases);

async function loadSidebarCases() {
  const res = await fetch(`/api/all_cases`);
  const data = await res.json();

  caseList.innerHTML = "";

  if (!data.cases || data.cases.length === 0) {
    caseSidebar.classList.add("d-none");
    return;
  }

  data.cases.forEach(item => {
    const entry = document.createElement("div");
    entry.className = "case-entry border-bottom py-2";
    entry.innerText = `Patient: ${item.patient_id} | Case: ${item.timestamp}`;
    entry.style.cursor = "pointer";
    entry.onclick = () => loadCase(item.patient_id, item.timestamp);
    caseList.appendChild(entry);
  });

  caseSidebar.classList.remove("d-none");
}


async function loadCase(patientId, timestamp) {
  const res = await fetch(`/api/load_case/${patientId}/${timestamp}`);
  const data = await res.json();

  activePatientId = patientId;
  activeTimestamp = timestamp;
  polygonAnnotations = data.annotations || [];

  inputImg = await loadImage("data:image/png;base64," + data.input_base64);
  mask1 = await loadImage("data:image/png;base64," + data.output1_base64);
  mask2 = await loadImage("data:image/png;base64," + data.output2_base64);

  document.getElementById("generalNotes").value = data.notes || "";

  document.getElementById("results").classList.remove("d-none");
  document.getElementById("controlsPanel").classList.remove("d-none");
  document.getElementById("legendPanel").classList.remove("d-none");
  document.getElementById("doctorNotesPanel").classList.remove("d-none");
  document.getElementById("exportBtn").classList.remove("d-none");
  document.getElementById("patientInfo").classList.remove("d-none");
  document.getElementById("saveBtn").classList.remove("d-none");
  document.getElementById("uploadSection").classList.add("d-none");
  document.getElementById("displayPatientId").textContent = patientId;
  document.getElementById("displayDoctorId").textContent = data.doctor_id || "N/A";

  console.log("Loaded case for patient %s at timestamp %s", patientId, timestamp, doctorId);

  const opacity1 = parseFloat(document.getElementById("opacity1").value);
  const opacity2 = parseFloat(document.getElementById("opacity2").value);
  const overlayColor1 = document.getElementById("overlayColor1").value;
  const overlayColor2 = document.getElementById("overlayColor2").value;

  const smooth1 = document.getElementById("smooth1").checked;
  const smooth2 = document.getElementById("smooth2").checked;

  drawCanvas("canvas1", inputImg, mask1, opacity1, "overlay", smooth1, overlayColor1);
  drawCanvas("canvas2", inputImg, mask2, opacity2, "overlay", smooth2, overlayColor2);
}

document.getElementById("newCaseButton").addEventListener("click", () => {
  window.location.href = "/";
});

