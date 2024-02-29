import numpy as np
import matplotlib.pyplot as plt

def func(x, y):
    """定义一个复杂的高度函数"""
    return np.sin(0.5 * x**2 - 0.25 * y**2 + 3) * np.cos(2 * x + 1 - np.exp(y))

def gradient(x, y):
    """计算复杂函数的梯度"""
    dx = 0.5 * x * np.cos(0.5 * x**2 - 0.25 * y**2 + 3) - 2 * np.sin(2 * x + 1 - np.exp(y))
    dy = -0.5 * y * np.cos(0.5 * x**2 - 0.25 * y**2 + 3) - np.exp(y) * np.sin(2 * x + 1 - np.exp(y))
    return dx, dy

def gradient_descent(start_x, start_y, learning_rate, num_iterations):
    """梯度下降法寻找最快下山路径"""
    x, y = start_x, start_y
    path = [(x, y)]
    for i in range(num_iterations):
        grad_x, grad_y = gradient(x, y)
        x, y = x - learning_rate * grad_x, y - learning_rate * grad_y
        path.append((x, y))
    return np.array(path)

# 参数设置
start_x, start_y = 0.3, 0.7  # 起始点
learning_rate = 0.01  # 学习率
num_iterations = 1000  # 迭代次数

# 执行梯度下降
path = gradient_descent(start_x, start_y, learning_rate, num_iterations)

# 绘制下山路径
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)
plt.contour(X, Y, Z, 50)
plt.plot(path[:, 0], path[:, 1], 'r*-')
plt.title("Non-linear Gradient Descent Path on a Complex Hill")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
