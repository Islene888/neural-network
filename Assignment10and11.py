import os
import numpy as np
import cv2
import tensorflow as tf

# Define functions for data augmentation
def horizontal_flip(image):
    return cv2.flip(image, 1)

def vertical_flip(image):
    return cv2.flip(image, 0)

# Define functions for data normalization
def normalize_image(image):
    return image / 255.0

# Model Definition
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Data Preparation
def prepare_data(dataset_dir, target_size, batch_size):
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []

    # Create a mapping from class names to integer labels
    class_names = sorted(os.listdir(dataset_dir))
    class_to_label = {class_name: i for i, class_name in enumerate(class_names)}

    # Iterate through the dataset directory to load images and labels
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        images = os.listdir(class_dir)
        num_images = len(images)
        # Use 70% of the images for training and 30% for validation
        num_train = int(0.7 * num_images)
        for i, image_name in enumerate(images):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)  # Resize images to target size
            image = normalize_image(image)
            label = class_to_label[class_name]  # Use the mapping to get the label
            if i < num_train:
                train_images.append(image)
                train_labels.append(label)
                # Augment data by flipping horizontally and vertically
                train_images.append(horizontal_flip(image))
                train_labels.append(label)
                train_images.append(vertical_flip(image))
                train_labels.append(label)
            else:
                validation_images.append(image)
                validation_labels.append(label)

    # Convert lists to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)

    # Shuffle training data
    shuffle_indices = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_indices]
    train_labels = train_labels[shuffle_indices]

    return train_images, train_labels, validation_images, validation_labels

# Train the model
def train_model(model, train_images, train_labels, validation_images, validation_labels, epochs):
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(validation_images, validation_labels)
    )
    return history

# Main function
def main():
    # Data directory
    dataset_dir = r"D:\Notets\python\Neurons\HW8_train_dataset"
    # Target image size
    target_size = (64, 64)
    # Batch size
    batch_size = 32
    # Number of epochs
    epochs = 10

    # Create model
    model = create_model(input_shape=target_size + (3,))
    # Compile model
    model = compile_model(model)
    # Prepare data
    train_images, train_labels, validation_images, validation_labels = prepare_data(dataset_dir, target_size, batch_size)
    # Train model
    history = train_model(model, train_images, train_labels, validation_images, validation_labels, epochs)
    # Optionally return or print the training history
    return history

if __name__ == "__main__":
    main()
