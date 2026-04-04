import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit

import time

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def coef_poly(coefficients, x):
    ttl = 0
    for i, coeff in enumerate(coefficients):
        ttl += coeff*x**(len(coefficients)-(i+1))
    return ttl

def r2_score_np(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

_true_coefficients = [
    4,
    -4,
    7,
    5,
    -3,
    -2
]

points = []

# Append 11 points at 0.2 intervals from [-1, 1]
intervals = 20
for i in range(intervals + 1):
    x = -1+(2/intervals)*i
    points.append((x, coef_poly(_true_coefficients, x)))

points = np.array(points)

x = np.array([point[0] for point in points], dtype=float)
y = np.array([point[1] for point in points], dtype=float)

scipy_time_start = time.perf_counter()
params, _t = curve_fit(cubic, x, y)
scipy_time_end = time.perf_counter()
print(f"Scipy Coefficient: {params}")
print(f"Scipy R^2: {r2_score_np(np.array(y), np.array([coef_poly(params, point[0]) for point in points]))}")
print(f"Time: {scipy_time_end-scipy_time_start}")

# latex_dataset_string = "D="
# for point in points:
#     latex_dataset_string = latex_dataset_string + f"({point[0]:.3f},{point[1]:.3f}),"
# 
# print(latex_dataset_string)
coefficients = [
    0,
    0,
    0,
    0
]
recent_grad = np.ones(len(coefficients))
coefficients = np.array(coefficients)

alpha = 0.001

def forward_prop():
    tss = 0
    for point in points:
        f_theta_x = coef_poly(coefficients, point[0])
        square_error = (point[1]-f_theta_x)**2
        tss += square_error
    return tss

def back_prop_sum_all():
    global recent_grad
    grad = np.zeros(len(coefficients))
    for x, y in points:
        err = (y - coef_poly(coefficients, x))
        grad += -2 * err * np.array([x**(len(coefficients)-(i+1)) for i in range(len(coefficients))])
    recent_grad = grad
    return grad

def minimize():
    global coefficients
    coefficients = coefficients - alpha*back_prop_sum_all()

epoch_count = 0
epoch_hist = []
time_hist = []
loss_hist = []
r2_hist = []
iters = []


powers_of_10 = 10 
for i in range(powers_of_10+1):
    iters.append(10**(-i))
    coefficients=np.zeros(len(coefficients))
    epoch_count = 0
    gd_time_start = time.perf_counter()
    while (np.any(abs(recent_grad) > 10**(-i))):
        minimize()
        epoch_count += 1
    loss_hist.append(forward_prop()/21)
    gd_time_end = time.perf_counter()
    epoch_hist.append(epoch_count)
    time_hist.append(gd_time_end-gd_time_start)
    r2_hist.append(r2_score_np(np.array([point[1] for point in points]), np.array([coef_poly(coefficients, point[0]) for point in points])))
print("Benchmark done")

# iters = np.arange(powers_of_10+1)
# iters *= -1
plt.subplot(2, 1, 1)
plt.plot(iters, time_hist, label=r'$t$')
plt.plot(iters, loss_hist, label=r'$\bar\mathcal{L}$')
plt.plot(iters, r2_hist, label=r'$R^2$')
plt.legend()
plt.xscale('log')
plt.ylabel(r'Tims (s) / Average Loss / $R^2$')
plt.subplot(2, 1, 2)
plt.plot(iters, epoch_hist, label=r'$I$')

plt.xscale('log')
plt.xlabel("Threshold")
plt.ylabel("Iterations")
# plt.title("Iterations against Thershold")
plt.legend()
plt.savefig("time_comp3")
plt.show()