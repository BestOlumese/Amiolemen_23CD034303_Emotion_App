from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformersEmotionDetector:
    def __init__(self):
        self.model_name = 'abhilash88/face-emotion-detection'
        self.processor = None
        self.model = None
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.load_model()
    
    def load_model(self):
        """Load the transformers model and processor"""
        try:
            logger.info("Loading emotion detection model...")
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTForImageClassification.from_pretrained(self.model_name)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """Convert OpenCV image to PIL format and preprocess"""
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            return pil_image
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise
    
    def detect_faces(self, image):
        """Simple face detection to crop the face region"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Return the largest face
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                face_roi = image[y:y+h, x:x+w]
                return face_roi
            else:
                return image  # Return original image if no face detected
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return image  # Return original image as fallback
    
    def predict_emotion(self, image):
        """Main method to predict emotion from image"""
        try:
            # First, try to detect and crop face
            processed_image = self.detect_faces(image)
            
            # Convert to PIL format
            pil_image = self.preprocess_image(processed_image)
            
            # Preprocess for the model
            inputs = self.processor(pil_image, return_tensors="pt")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
            
            # Get results
            predicted_emotion = self.emotions[predicted_class]
            confidence = predictions[0][predicted_class].item()
            
            logger.info(f"Predicted: {predicted_emotion} (Confidence: {confidence:.3f})")
            return predicted_emotion, float(confidence)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return neutral as fallback
            return "Neutral", 0.5

# Create global instance
try:
    emotion_detector = TransformersEmotionDetector()
    logger.info("Emotion detector initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize emotion detector: {e}")
    emotion_detector = None