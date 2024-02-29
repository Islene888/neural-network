import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers


dataset_dir = r"D:\Notets\python\Neurons\HW8_train_dataset"

# Model definition
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

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

test_dataset_dir = r"D:\Notets\python\Neurons\HW8_test_dataset"  # 测试数据集路径

test_datagen = ImageDataGenerator(rescale=1./255)  # 仅用于重新缩放图像，无需验证分割

test_generator = test_datagen.flow_from_directory(
    test_dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse')  # 测试数据，注意确保 class_mode 与之前相同

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

