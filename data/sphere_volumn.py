import numpy as np
from scipy.special import gamma

pi = np.pi
R = 5
r  =4
shell = []
for n in range(2, 1000):
    gamma_n_2 = gamma(n/2 + 1)
    volume1 = (pi**(n/2)) / gamma_n_2 * (R**n)
    volume2 = (pi**(n/2)) / gamma_n_2 * (r**n)
    delta = volume1 - volume2
    shell.append(delta)
    print(f"{n} : {volume1} - {volume2} = {delta}")

# plot the curve of shell
import matplotlib.pyplot as plt
plt.plot(np.arange(2, 1000), shell, '*', color='black', markersize=10)
plt.xlabel('dimension')
plt.ylabel('volume of shell')
plt.savefig(f'./figures/shell-volume.png')