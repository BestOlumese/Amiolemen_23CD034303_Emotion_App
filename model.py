import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

class EmotionDetector:
    def __init__(self, model_path='trained_model.h5'):
        self.model_path = model_path
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.model = self.load_model()
    
    def load_model(self):
        """Load pre-trained model or create a new one"""
        try:
            model = tf.keras.models.load_model(self.model_path)
            print("Pre-trained model loaded successfully")
        except:
            print("Creating new model architecture...")
            model = self.create_model()
        return model
    
    def create_model(self):
        """Create model architecture (this would be your trained model)"""
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to 48x48
        gray = cv2.resize(gray, (48, 48))
        
        # Normalize pixel values
        gray = gray / 255.0
        
        # Reshape for model input
        gray = gray.reshape(1, 48, 48, 1)
        
        return gray
    
    def predict_emotion(self, image):
        """Predict emotion from image"""
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image)
        emotion_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return self.emotions[emotion_index], float(confidence)

# Create a global detector instance
emotion_detector = EmotionDetector()