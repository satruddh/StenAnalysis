from datetime import datetime
from flask import Flask, request, jsonify
import os
import numpy as np
import io
import base64
from PIL import Image
import torch
from utils import preprocess_image, run_inference
import zipfile
import json
from fpdf import FPDF
from flask import send_file
from flask import render_template


app = Flask(__name__, static_folder="static", template_folder="templates")
UPLOAD_FOLDER = "uploads"
BASE_CASE_DIR = "cases"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_CASE_DIR,exist_ok=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def tensor_to_base64(tensor):
    """Convert a PyTorch tensor to a base64 image string."""
    tensor = tensor.squeeze().cpu()  # Remove any single-dimensional entries

    # Handle 3-channel (RGB) or 1-channel (Grayscale)
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        # (3, H, W) to (H, W, 3)
        tensor = tensor.permute(1, 2, 0).numpy()
        mode = "RGB"
    elif tensor.ndim == 2:
        tensor = tensor.numpy()
        mode = "L"
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    # Convert to uint8
    tensor = (tensor * 255).astype(np.uint8)

    # Create the PIL image
    image = Image.fromarray(tensor, mode=mode)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route("/api/inference", methods=["POST"])
def inference():
    if "image" not in request.files or "patient_id" not in request.form:
        return jsonify({"error": "Image and Patient ID are required"}), 400

    file = request.files["image"]
    patient_id = request.form["patient_id"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    patient_dir = os.path.join(BASE_CASE_DIR, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    # Save the original uploaded image
    input_filename = f"{timestamp}_input.png"
    file_path = os.path.join(patient_dir, input_filename)
    file.save(file_path)

    with open(file_path,'rb') as f:
        input_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Run inference

    try:
        image_tensor = preprocess_image(file_path)
        output1, output2 = run_inference(image_tensor)

        # Convert and save outputs
        output1_base64 = tensor_to_base64(output1)
        output2_base64 = tensor_to_base64(output2)

        with open(os.path.join(patient_dir, f"{timestamp}_output1.png"), "wb") as f:
            f.write(base64.b64decode(output1_base64))
    
        with open(os.path.join(patient_dir, f"{timestamp}_output2.png"), "wb") as f:
            f.write(base64.b64decode(output2_base64))
    
        return jsonify({
            "message": "Inference successful",
            "output1_base64": output1_base64,
            "output2_base64": output2_base64,
            "input_base64": input_base64,
            "timestamp": timestamp
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/export", methods=["POST"])
def export_analysis():
    try:
        data = request.get_json()
        patient_id = data["patient_id"]
        timestamp = data["timestamp"]
        doctor_id = data["doctor_id"]
        notes = data["notes"]
        annotations = data["annotations"]
        annotated_images = data["images"]

        export_dir = os.path.join(BASE_CASE_DIR, patient_id, timestamp)
        os.makedirs(export_dir, exist_ok=True)

        # Save annotated images
        saved_paths = []
        for name, base64_str in annotated_images.items():
            img_data = base64.b64decode(base64_str.split(",")[1])
            path = os.path.join(export_dir, f"{name}.png")
            with open(path, "wb") as f:
                f.write(img_data)
            saved_paths.append(path)

        # Save notes.txt
        notes_path = os.path.join(export_dir, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(notes)

        # Save annotations.json
        annotation_path = os.path.join(export_dir, "annotations.json")
        with open(annotation_path, "w") as f:
            json.dump(annotations, f, indent=2)

        # Create a PDF report
        pdf_path = os.path.join(export_dir, "report.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Stenosis Segmentation Analysis Report", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Doctor ID: {doctor_id}", ln=True)
        pdf.cell(0, 10, f"Patient ID: {patient_id}", ln=True)
        pdf.cell(0, 10, f"Timestamp: {timestamp}", ln=True)
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Doctor's Notes:\n{notes}")
        pdf.ln(10)
        
        # Embed annotated images
        for name in annotated_images.keys():
            img_path = os.path.join(export_dir, f"{name}.png")
            if os.path.exists(img_path):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"{name.replace('_', ' ').capitalize()}:", ln=True)
                pdf.image(img_path, w=120)  # You can adjust width as needed
                pdf.ln(10)
        
        pdf.output(pdf_path)

        # Create ZIP bundle
        zip_path = os.path.join(export_dir, "export_bundle.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(notes_path, "notes.txt")
            zipf.write(annotation_path, "annotations.json")
            zipf.write(pdf_path, "report.pdf")
            for path in saved_paths:
                zipf.write(path, os.path.basename(path))

        return send_file(zip_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
