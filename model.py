# model.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# CONFIG
IMG_SIZE = (48, 48)  # common for FER
BATCH_SIZE = 64
EPOCHS = 40
DATA_DIR = "data"  # expects data/train and data/val folders
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")
NUM_CLASSES = None  # inferred

def build_model(input_shape=(48,48,1), num_classes=7):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise Exception("Please create data/train and data/test folders with emotion subfolders.")

    # We'll use grayscale images
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    global NUM_CLASSES
    NUM_CLASSES = train_gen.num_classes
    print("Detected classes:", train_gen.class_indices)

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=NUM_CLASSES)

    os.makedirs(MODEL_DIR, exist_ok=True)

    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early]
    )

    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
