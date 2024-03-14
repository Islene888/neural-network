import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(10,)),  # Input layer with 10 features
        Dense(8, activation='relu'),  # Second layer
        Dense(8, activation='relu'),  # Third layer
        Dense(4, activation='relu'),  # Fourth layer
        Dense(1, activation='sigmoid')  # Output layer
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def generate_dataset(num_samples=1000, num_features=10):
    np.random.seed(42)
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(2, size=num_samples)
    return X, y


if __name__ == "__main__":
    model = create_model()

    X_train, y_train = generate_dataset()

    model.fit(X_train, y_train, epochs=10, validation_split=0.2)

    model.summary()