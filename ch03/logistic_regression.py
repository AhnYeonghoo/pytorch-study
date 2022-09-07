import numpy as np
import matplotlib.pyplot as plt
import os

# definition Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y, 'g')
# plt.plot([0,0], [1.0, 0.0], ':')
# plt.title('Sigmoid Function')
# plt.show()

# y1 = sigmoid(0.5*x)
# y2 = sigmoid(x)
# y3 = sigmoid(2*x)

# plt.plot(x, y1, 'r', linestyle='--')
# plt.plot(x, y2 ,'g')
# plt.plot(x, y3, 'b', linestyle='--')
# plt.plot([0,0], [1.0, 0.0], ':')
# plt.title('Sigmoid Function')
# plt.show()

y1 = sigmoid(0.5+x)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0,0], [1.0, 0.0], ':')
plt.title('Sigmoid Function')
plt.show()