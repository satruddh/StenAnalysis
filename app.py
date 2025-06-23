from datetime import datetime
from flask import Flask, request, jsonify
import os
import numpy as np
import io
import base64
from PIL import Image
import torch
from utils import convert_dicom_to_png, preprocess_image, run_inference
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
    tensor = tensor.squeeze().cpu()  

    if tensor.ndim == 3 and tensor.shape[0] == 3:
        
        tensor = tensor.permute(1, 2, 0).numpy()
        mode = "RGB"
    elif tensor.ndim == 2:
        tensor = tensor.numpy()
        mode = "L"
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    
    tensor = (tensor * 255).astype(np.uint8)

    
    image = Image.fromarray(tensor, mode=mode)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route("/api/inference", methods=["POST"])
def inference():
    if "image" not in request.files or "patient_id" not in request.form or "input_type" not in request.form:
        return jsonify({"error": "Image, Patient ID, and Input Type are required"}), 400

    file = request.files["image"]
    patient_id = request.form["patient_id"]
    input_type = request.form["input_type"].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    patient_dir = os.path.join(BASE_CASE_DIR, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    info_path = os.path.join(patient_dir, "patient_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)

    info.update({
        "patient_id": patient_id,
        "name": request.form.get("name", ""),
        "age": request.form.get("age", ""),
        "weight": request.form.get("weight", ""),
        "sex": request.form.get("sex", ""),
        "bg": request.form.get("bg", "")
    })

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    case_dir = os.path.join(patient_dir, f"case_{timestamp}")
    os.makedirs(case_dir, exist_ok=True)

    input_png_path = os.path.join(case_dir, "input.png")

    if input_type == "png":
        file.save(input_png_path)
    elif input_type == "dicom":
        dicom_path = os.path.join(case_dir, "input.dcm")
        file.save(dicom_path)

        try:
            convert_dicom_to_png(
                dicom_file_path=dicom_path,
                output_png_path=input_png_path,
                metadata_output_path=os.path.join(case_dir, "dicom_metadata.json")
            )
        except Exception as e:
            return jsonify({"error": f"Failed to process DICOM: {str(e)}"}), 500
    else:
        return jsonify({"error": "Unsupported input type"}), 400

    
    
    

    with open(input_png_path,'rb') as f:
        input_base64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        image_tensor = preprocess_image(input_png_path)
        output1, output2 = run_inference(image_tensor)

        
        output1_base64 = tensor_to_base64(output1)
        output2_base64 = tensor_to_base64(output2)

        print

        with open(os.path.join(case_dir, f"{timestamp}_output1.png"), "wb") as f:
            f.write(base64.b64decode(output1_base64))
    
        with open(os.path.join(case_dir, f"{timestamp}_output2.png"), "wb") as f:
            f.write(base64.b64decode(output2_base64))
    
        print(f"Inference completed for patient {patient_id} at {timestamp}")
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

        patient_dir = os.path.join(BASE_CASE_DIR, patient_id)
        case_dir = os.path.join(patient_dir, f"case_{timestamp}")
        os.makedirs(case_dir, exist_ok=True)

        
        saved_paths = []
        for name, base64_str in annotated_images.items():
            img_data = base64.b64decode(base64_str.split(",")[1])
            path = os.path.join(case_dir, f"{name}.png")
            with open(path, "wb") as f:
                f.write(img_data)
            saved_paths.append(path)

        
        notes_path = os.path.join(case_dir, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(notes)

        
        annotation_path = os.path.join(case_dir, "annotations.json")
        with open(annotation_path, "w") as f:
            json.dump(annotations, f, indent=2)

        
        pdf_path = os.path.join(case_dir, "report.pdf")
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
        
        
        for name in annotated_images.keys():
            img_path = os.path.join(case_dir, f"{name}.png")
            if os.path.exists(img_path):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"{name.replace('_', ' ').capitalize()}:", ln=True)
                pdf.image(img_path, w=120)  
                pdf.ln(10)
        
        pdf.output(pdf_path)

        
        zip_path = os.path.join(case_dir, "export_bundle.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(notes_path, "notes.txt")
            zipf.write(annotation_path, "annotations.json")
            zipf.write(pdf_path, "report.pdf")
            for path in saved_paths:
                zipf.write(path, os.path.basename(path))

        return send_file(zip_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/save", methods=["POST"])
def save_analysis():
    try:
        data = request.get_json()
        patient_id = data["patient_id"]
        timestamp = data["timestamp"]
        doctor_id = data["doctor_id"]
        notes = data["notes"]
        annotations = data["annotations"]
        annotated_images = data["images"]

        patient_dir = os.path.join(BASE_CASE_DIR, patient_id)
        case_dir = os.path.join(patient_dir, f"case_{timestamp}")
        os.makedirs(case_dir, exist_ok=True)

        
        notes_path = os.path.join(case_dir, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(notes)

        
        annotation_path = os.path.join(case_dir, "annotations.json")
        with open(annotation_path, "w") as f:
            json.dump(annotations, f, indent=2)

        
        annotated_files = {}
        for name, base64_str in annotated_images.items():
            img_data = base64.b64decode(base64_str.split(",")[1])
            filename = f"{name}.png"
            img_path = os.path.join(case_dir, filename)
            with open(img_path, "wb") as f:
                f.write(img_data)
            annotated_files[name] = filename

        
        index_path = os.path.join(patient_dir, "index.json")
        index_data = {"patient_id": patient_id, "cases": []}
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index_data = json.load(f)

        
        index_data["cases"] = [
            entry for entry in index_data["cases"]
            if not (entry["timestamp"] == timestamp and entry["doctor_id"] == doctor_id)
        ]

        index_data["cases"].append({
            "timestamp": timestamp,
            "doctor_id": doctor_id,
            "notes": notes,
            "annotations": "annotations.json",
            "annotated_images": annotated_files
        })

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        return jsonify({"message": "Analysis saved successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cases/<patient_id>")
def list_cases(patient_id):
    patient_dir = os.path.join(BASE_CASE_DIR, patient_id)
    if not os.path.exists(patient_dir):
        return jsonify({"cases": []})

    cases = []
    for folder in sorted(os.listdir(patient_dir), reverse=True):
        if not folder.startswith("case_"):
            continue
        case_path = os.path.join(patient_dir, folder)
        notes_path = os.path.join(case_path, "notes.txt")
        note = ""
        if os.path.exists(notes_path):
            with open(notes_path) as f:
                note = f.read().strip().split("\n")[0]  
        cases.append({
            "timestamp": folder.replace("case_", ""),
            "note": note
        })

    return jsonify({"cases": cases})


@app.route("/api/load_case/<patient_id>/<timestamp>")
def load_case(patient_id, timestamp):
    case_dir = os.path.join(BASE_CASE_DIR, patient_id, f"case_{timestamp}")
    
    input_path = os.path.join(case_dir, "input.png")
    output1_path = os.path.join(case_dir, "%s_output1.png"%timestamp)
    output2_path = os.path.join(case_dir, "%s_output2.png"%timestamp)
    notes_path = os.path.join(case_dir, "notes.txt")
    annotations_path = os.path.join(case_dir, "annotations.json")
    index_path = os.path.join(BASE_CASE_DIR, patient_id, "index.json")
    info_path = os.path.join(BASE_CASE_DIR, patient_id, "patient_info.json")

    doctor_id = ""
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = json.load(f)
            for entry in data.get("cases", []):
                if entry["timestamp"] == timestamp:
                    doctor_id = entry.get("doctor_id", "")
                    break

    def encode_image(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    response = {
        "input_base64": encode_image(input_path),
        "output1_base64": encode_image(output1_path),
        "output2_base64": encode_image(output2_path),
        "notes": "",
        "annotations": [],
        "timestamp": timestamp,
        "doctor_id": doctor_id
    }

    if os.path.exists(info_path):
        with open(info_path) as f:
            response["patient_info"] = json.load(f)

    if os.path.exists(notes_path):
        with open(notes_path) as f:
            response["notes"] = f.read()

    if os.path.exists(annotations_path):
        with open(annotations_path) as f:
            response["annotations"] = json.load(f)

    return jsonify(response)

@app.route("/api/all_cases")
def all_cases():
    all_data = []
    for patient_id in os.listdir(BASE_CASE_DIR):
        patient_path = os.path.join(BASE_CASE_DIR, patient_id)
        index_path = os.path.join(patient_path, "index.json")
        if not os.path.exists(index_path):
            continue
        try:
            with open(index_path) as f:
                index = json.load(f)
                for case in index.get("cases", []):
                    all_data.append({
                        "patient_id": patient_id,
                        "timestamp": case["timestamp"],
                        "note": case.get("notes", "").split("\n")[0]  
                    })
        except Exception as e:
            print(f"Error reading index.json for {patient_id}: {e}")
    return jsonify({"cases": sorted(all_data, key=lambda x: x["timestamp"], reverse=True)})



@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
