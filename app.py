from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import os
import logging
from PIL import Image as PILImage
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import emotion detector
try:
    from model import emotion_detector
    logger.info("Transformers emotion detector loaded successfully!")
except ImportError as e:
    logger.error(f"Failed to import emotion detector: {e}")
    emotion_detector = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize database
def init_db():
    conn = sqlite3.connect('emotion_detection.db', check_same_thread=False)
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
    conn = sqlite3.connect('emotion_detection.db', check_same_thread=False)
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

@app.route('/health')
def health_check():
    """Check if the emotion detector is working"""
    if emotion_detector:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'message': 'Emotion detector is ready'
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'message': 'Emotion detector failed to load'
        }), 500

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        name = request.form.get('name', 'Anonymous')
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image file
        image_bytes = image_file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        logger.info(f"Processing image for {name}, shape: {image.shape}")
        
        # Detect emotion
        if emotion_detector is None:
            return jsonify({
                'error': 'Emotion detection service unavailable',
                'emotion': 'Neutral',
                'confidence': 50.0
            }), 503
        
        emotion, confidence = emotion_detector.predict_emotion(image)
        
        # Save processed image
        image_dir = 'static/uploads'
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.replace(' ', '_')}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        
        # Save the image
        cv2.imwrite(image_path, image)
        
        # Save to database
        save_to_db(name, image_path, emotion, confidence, True)
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'image_path': image_path,
            'name': name
        })
        
    except Exception as e:
        logger.error(f"Error in detect_emotion: {str(e)}")
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'emotion': 'Neutral',
            'confidence': 50.0
        }), 500

@app.route('/live_feed')
def live_feed():
    return render_template('live_feed.html')

@app.route('/video_feed')
def video_feed():
    """Live video feed endpoint"""
    try:
        camera = cv2.VideoCapture(0)
        
        def generate_frames():
            while True:
                success, frame = camera.read()
                if not success:
                    break
                else:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        return Response(generate_frames(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    except Exception as e:
        logger.error(f"Video feed error: {e}")
        return "Camera not available", 503

@app.route('/history')
def history():
    conn = sqlite3.connect('emotion_detection.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        SELECT id, name, image_path, emotion, confidence, timestamp 
        FROM detection_history 
        ORDER BY timestamp DESC 
        LIMIT 50
    ''')
    history_data = c.fetchall()
    conn.close()
    
    return render_template('history.html', history=history_data)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear detection history"""
    try:
        conn = sqlite3.connect('emotion_detection.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('DELETE FROM detection_history')
        conn.commit()
        conn.close()
        
        # Also clear uploaded images
        image_dir = 'static/uploads'
        if os.path.exists(image_dir):
            for file in os.listdir(image_dir):
                if file.endswith('.jpg'):
                    os.remove(os.path.join(image_dir, file))
        
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    
    # Check if model is loaded
    if emotion_detector:
        logger.info("🚀 Starting Flask app with emotion detection")
    else:
        logger.warning("⚠️ Starting Flask app WITHOUT emotion detection")
    
    app.run(debug=True, host='0.0.0.0', port=5000)