import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成更有代表性的数据集
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)  # 调整标签的形状以符合模型的输入

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义模型
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)),  # 输入层
    Dropout(0.1),  # Dropout层减少过拟合
    Dense(8, activation='relu'),  # 隐藏层
    Dense(8, activation='relu'),  # 隐藏层
    Dense(4, activation='relu'),  # 隐藏层
    Dense(1, activation='sigmoid')  # 输出层
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# 模型摘要
model.summary()

# 评估模型
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))
