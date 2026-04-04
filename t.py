 #for i in range(10):
 #    print(i)

import numpy as np

x = np.array([0, 2, 3, 4])
# print(x.size)

import numpy as np
import matplotlib.pyplot as plt

_true_coefficients = [
    4,
    -4,
    7,
    5,
    -3,
    -2
]

points = []
for i in range(11):
    points.append(-1+0.2*i, )

points = np.array(points)
# print(len(points.shape))

# coefficients = [
#     0,
#     0,
#     0,
#     0
# ]

# coefficients = np.array(coefficients)

alpha = 0.001

def coef_poly(coefficients, x):
    ttl = 0
    for i, coeff in enumerate(coefficients):
        ttl += coeff*x**(len(coefficients)-(i+1))
    return ttl

coefficients = [
    2,
    1
]
print(coef_poly(_true_coefficients, -0.48429))
# print(len(coefficients))