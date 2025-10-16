from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

# ---------------------------
# Flask App Initialization
# ---------------------------
app = Flask(__name__)
app.secret_key = "secure_key"  # For flash messages

# ---------------------------
# Configuration
# ---------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ---------------------------
# Load the Pretrained Model Once
# ---------------------------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

CLASS_LABELS = ["pituitary", "glioma", "notumor", "meningioma"]

# ---------------------------
# Utility Functions
# ---------------------------
def allowed_file(filename):
    """Check file extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess the image for prediction."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_tumor(image_path):
    """Make prediction and return label + confidence."""
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    label = CLASS_LABELS[predicted_idx]
    return ("No Tumor" if label == "notumor" else f"Tumor: {label}"), confidence

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Home page with upload form."""
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            flash("No file selected!")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Invalid file format. Please upload JPG or PNG images.")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        result, confidence = predict_tumor(file_path)
        return render_template(
            "index.html",
            result=result,
            confidence=f"{confidence * 100:.2f}%",
            file_path=url_for("get_uploaded_file", filename=filename),
        )

    return render_template("index.html", result=None)

@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    """Serve uploaded images securely."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
