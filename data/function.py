import numpy as np
import matplotlib.pyplot as plt
from math import log, exp

X = np.linspace(1e-7, 0.1, 100000)  # Range of X values
# Y = np.array([-1.0/(0.01*log(x)) for x in X])
# Y = -1.0/(log(X))  # Calculate Y values

Y = np.exp(20 * X)

# fit a line uses X and Y
# z = np.polyfit(X, Y, 1)
# p = np.poly1d(z)
# print(f'p = {p}')
# # the line equation:
# print(f'y = {z[0]} * x + {z[1]}')

# plot the line, too
# plt.plot(X, p(X), "r--")


plt.plot(X, Y)
plt.xlabel('X')
plt.ylabel('f(X) = -1/ln(x)')
# plt.title('Plot of f(X) = exp(X) / X')
plt.grid(True)
plt.savefig("./data/test.png")
print(f'save to ./data/test.png')