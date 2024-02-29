# --- Imports ---
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers

# --- Dataset Directory ---
dataset_dir = r"D:\Notets\python\Neurons\HW8_train_dataset"

# --- Model Definition ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Data Preparation ---
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)  # using split for training and validation

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse',
    subset='training')  # Training data

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse',
    subset='validation')  # Validation data

# Train the model with the training and validation generators
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Functions for data normalization 
def normalize_inputs(inputs):
    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0)
    return (inputs - mean) / std

def min_max_normalize_inputs(inputs):
    min_val = np.min(inputs, axis=0)
    max_val = np.max(inputs, axis=0)
    return (inputs - min_val) / (max_val - min_val)