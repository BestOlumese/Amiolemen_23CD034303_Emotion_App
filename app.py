from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import os
from model import emotion_detector
import base64

app = Flask(__name__)

# Initialize database
def init_db():
    conn = sqlite3.connect('emotion_detection.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detection_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  image_path TEXT,
                  emotion TEXT,
                  confidence REAL,
                  timestamp DATETIME,
                  is_online BOOLEAN)''')
    conn.commit()
    conn.close()

init_db()

def save_to_db(name, image_path, emotion, confidence, is_online):
    conn = sqlite3.connect('emotion_detection.db')
    c = conn.cursor()
    c.execute('''INSERT INTO detection_history 
                 (name, image_path, emotion, confidence, timestamp, is_online)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (name, image_path, emotion, confidence, datetime.now(), is_online))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        name = request.form.get('name', 'Anonymous')
        image_file = request.files['image']
        
        # Convert image to numpy array
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Detect emotion
        emotion, confidence = emotion_detector.predict_emotion(image)
        
        # Save image
        image_dir = 'static/uploads'
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        cv2.imwrite(image_path, image)
        
        # Save to database
        save_to_db(name, image_path, emotion, confidence, True)
        
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'image_path': image_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/live_feed')
def live_feed():
    return render_template('live_feed.html')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect emotion in frame
            emotion, confidence = emotion_detector.predict_emotion(frame)
            
            # Add emotion text to frame
            cv2.putText(frame, f'{emotion} ({confidence*100:.1f}%)', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def history():
    conn = sqlite3.connect('emotion_detection.db')
    c = conn.cursor()
    c.execute('SELECT * FROM detection_history ORDER BY timestamp DESC LIMIT 50')
    history_data = c.fetchall()
    conn.close()
    
    return render_template('history.html', history=history_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)