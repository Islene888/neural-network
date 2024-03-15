from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
(X, y), (X_test, y_test) = mnist.load_data()

# Filter out only the images for digits 8 and 9 for both training and test datasets
filter_indices = (y == 8) | (y == 9)
X, y = X[filter_indices], y[filter_indices] - 8  # Adjust labels to be 0 and 1
filter_indices_test = (y_test == 8) | (y_test == 9)
X_test, y_test = X_test[filter_indices_test], y_test[filter_indices_test] - 8  # Adjust labels to be 0 and 1

# Normalize the images from 0-255 to 0-1
X = X / 255.0
X_test = X_test / 255.0

# Flatten the images for the MLP (multilayer perceptron)
X = X.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the binary classification model
model = Sequential([
    Flatten(input_shape=(28*28,)),
    Dense(10, activation='relu'),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# EarlyStopping callback to stop training when the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
