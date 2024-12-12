# -*- coding: utf-8 -*-
"""LR_4_Task_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hy3CXYRgE8KUMy20PJ1D2l31UcIRE5j9
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 5, 10, 15, 20, 25])
y = np.array([21, 39, 51, 63, 70, 90])

n = len(x)
x_sum = x.sum()
y_sum = y.sum()
xy_sum = np.dot(x, y)
x_squared_sum = np.dot(x, x)

b = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum**2)
a = (y_sum - b * x_sum) / n

x_range = np.linspace(x.min(), x.max(), 100)
y_range = b * x_range + a

plt.figure(figsize=(8, 6))
plt.scatter(x, y, c="orange", label="Виміряні дані")
plt.plot(x_range, y_range, c="green", label=f"Регресійна лінія: Y = {a:.2f} + {b:.2f}X")
plt.xlabel("Змінна X")
plt.ylabel("Змінна Y")
plt.title("Лінійна регресія: метод найменших квадратів")
plt.legend()
plt.grid()
plt.show()

y_estimated = b * x + a
squared_errors = np.sum((y - y_estimated)**2)

print(f"Коефіцієнт нахилу (b): {b:.2f}")
print(f"Перетин (a): {a:.2f}")
print(f"Рівняння: Y = {a:.2f} + {b:.2f}X")
print(f"Сума квадратів помилок: {squared_errors:.2f}")