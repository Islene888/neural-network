import tensorflow as tf
import numpy as np
import cv2
import os

# # Load the MNIST dataset
# mnist = tf.keras.datasets.mnist
# (train_images, train_labels), (_, _) = mnist.load_data()
#
# # Create a directory to store the images
# output_dir = "mnist_images"
# os.makedirs(output_dir, exist_ok=True)
#
# # Resize the images to 20x20 and save 10 images for each number
# for num in range(10):
#     num_indices = np.where(train_labels == num)[0][:10]  # Get indices of the first 10 occurrences of the number
#     for i, index in enumerate(num_indices):
#         image = train_images[index]
#
#         # Resize the image to 20x20
#         resized_image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_AREA)
#
#         # Save the resized image
#         image_filename = os.path.join(output_dir, f"{num}_{i}.png")
#         cv2.imwrite(image_filename, resized_image)
#
#         print(f"Generated: {image_filename}")
#
# print("All images generated successfully.")
# # Close OpenCV windows
# cv2.destroyAllWindows()

import numpy as np
import os
from PIL import Image
import os
import numpy as np
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 载入图像数据
data_path = "D:/Notets/python/Neurons0/Assisgment/mnist_images"

image_size = (20, 20)
num_classes = 10


# 加载和预处理数据
def load_data(data_path):
    images = []
    labels = []
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            label = int(filename.split("_")[0])
            image_path = os.path.join(data_path, filename)
            try:
                image = Image.open(image_path).convert('L')
                image = image.resize(image_size)
                image_data = np.array(image) / 255.0
                images.append(image_data)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    return np.array(images), np.array(labels)


# 加载数据集
images, labels = load_data(data_path)

# 将标签转换为one-hot编码
labels = to_categorical(labels, num_classes=num_classes)

# 将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 构建卷积神经网络模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train.reshape(-1, 20, 20, 1), y_train, epochs=10, batch_size=32,
          validation_data=(x_test.reshape(-1, 20, 20, 1), y_test))

# 评估模型
test_loss, test_acc = model.evaluate(x_test.reshape(-1, 20, 20, 1), y_test, verbose=2)
print('\nTest accuracy:', test_acc)


# 预测
def predict_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize(image_size)
    image_data = np.array(image) / 255.0
    prediction = model.predict(np.array([image_data.reshape(20, 20, 1)]))
    return np.argmax(prediction)


# 示例预测
test_image_path = "D:/Notets/python/Neurons0/Assisgment/mnist_images/5_5.png"  # 示例测试图像路径
prediction = predict_image(test_image_path)
print("Predicted Label:", prediction)
