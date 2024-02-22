import tensorflow as tf
from benchmarks._bench.eigenpro_plot_mnist import x_train, y_train
from tensorflow.keras import layers, models, regularizers



input_shape = (784,)  # Example
num_classes = 10  # Example


model = models.Sequential([
    # Input layer with L2 regularization
    layers.Dense(128, activation='relu', input_shape=input_shape,
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),  
    
    # Hidden layer with L1 regularization
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l1(0.01)),
    layers.Dropout(0.5),  # Another dropout layer with 50% dropout rate

    # Output layer
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(x_val, y_val))

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=2)
print(f'Validation accuracy: {val_acc}, Validation loss: {val_loss}')
