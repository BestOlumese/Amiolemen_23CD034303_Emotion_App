# app.py
import os
import io
import base64
import sqlite3
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# CONFIG
MODEL_PATH = "models/emotion_model.h5"
UPLOAD_FOLDER = "uploads"
DB_PATH = "database/users.db"
IMG_SIZE = (48, 48)  # must match model

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once
model = None
emotion_labels = None

def init_model():
    global model, emotion_labels
    model = load_model(MODEL_PATH)
    # infer classes from model output length; client should match labels order.
    # You should preserve the same label order used during training.
    # Example mapping (update to the exact mapping in your training):
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    # If your training generator used alphabetical class order, ensure this list matches train_gen.class_indices order.

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            image_path TEXT,
            predicted_emotion TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(name, image_path, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (name, timestamp, image_path, predicted_emotion, confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, datetime.utcnow().isoformat(), image_path, prediction, float(confidence)))
    conn.commit()
    conn.close()

def preprocess_image_pil(pil_img):
    # convert to grayscale, resize, normalize
    img = pil_img.convert("L").resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch
    if arr.shape[-1] == 1:
        pass
    return arr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Accepts either file upload (multipart/form-data) with key 'image'
    # OR JSON with base64 frame 'image' and optional 'name' field
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    name = request.values.get('name', 'Anonymous')

    # Get image from form or json
    if 'image' in request.files:
        file = request.files['image']
        pil_img = Image.open(file.stream)
    else:
        data = request.get_json() or {}
        img_b64 = data.get('image', None) or request.form.get('image', None)
        if img_b64 is None:
            return jsonify({'error': 'No image provided'}), 400
        if img_b64.startswith('data:image'):
            # remove header
            header, img_b64 = img_b64.split(',', 1)
        try:
            img_data = base64.b64decode(img_b64)
            pil_img = Image.open(io.BytesIO(img_data))
        except Exception as e:
            return jsonify({'error': 'Invalid image data', 'details': str(e)}), 400

    img_arr = preprocess_image_pil(pil_img)  # shape (1, h, w, 1)

    # If model expects channels last with shape (1,48,48,1), ensure correct dims
    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr)
    confidence = float(np.max(preds))
    idx = int(np.argmax(preds))
    predicted_emotion = emotion_labels[idx] if emotion_labels else str(idx)

    # Save image to disk
    safe_name = name.replace(" ", "_")[:50]
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{safe_name}_{ts}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pil_img.convert("RGB").save(filepath)

    # Save to DB
    try:
        save_prediction(name, filepath, predicted_emotion, confidence)
    except Exception as e:
        print("DB save error:", e)

    return jsonify({
        'predicted_emotion': predicted_emotion,
        'confidence': confidence,
        'image_path': filepath
    })

if __name__ == '__main__':
    init_db()
    init_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
