import numpy as np
import json

n = 10000
sigma_z1 = 5
sigma_z0 = 3

z = np.random.binomial(1, 0.5, n)
x = np.zeros(n)
t = np.zeros(n)
y = np.zeros(n)

res = []

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

for i in range(n):
    x[i] = np.random.normal(z[i], sigma_z1 * z[i] + sigma_z0 * (1 - z[i]))
    t[i] = np.random.binomial(1, 0.75 * z[i] + 0.25 * (1 - z[i]))
    y[i] = np.random.binomial(1, sigmoid(3 * (z[i] + 2 * (2 * t[i] - 1))))

    res.append([int(z[i]), int(x[i]), int(t[i]), int(y[i])])

with open("synthetic_data.json", "w") as file:
   json.dump(res, file, indent=4)