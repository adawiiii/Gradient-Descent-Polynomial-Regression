import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit

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

params, _t = curve_fit(cubic, x, y)
print(f"Scipy Coefficient: {params}")
print(f"Scipy R^2: {r2_score_np(np.array(y), np.array([coef_poly(params, point[0]) for point in points]))}")

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
theta3_hist = []
theta2_hist = []
theta1_hist = []
theta0_hist = []
avg_loss_hist = []

while (np.any(abs(recent_grad) > 1e-2)):
    avg_loss_hist.append(forward_prop()/21)
    theta3_hist.append(coefficients[0])
    theta2_hist.append(coefficients[1])
    theta1_hist.append(coefficients[2])
    theta0_hist.append(coefficients[3])
    minimize()
    epoch_count += 1
iters = np.arange(epoch_count)

poly_string = ""
for i, coeff in enumerate(coefficients):
    poly_string = poly_string + f"{"+" if i!=0 else ""}({coeff:.2f}x^{(len(coefficients)-(i+1))})"
print("------------------")
print(f"GD Poly: {poly_string}")
print(f"GD Coefficients: {coefficients}")
print(f"GD R^2: {r2_score_np(np.array([point[1] for point in points]), np.array([coef_poly(coefficients, point[0]) for point in points]))}")
print(f"Epochs: {epoch_count}")
print(f"Most recent gradient vector: {recent_grad}")

plt.plot(iters, theta3_hist, label=r'$\theta_3$')
plt.plot(iters, theta2_hist, label=r'$\theta_2$')
plt.plot(iters, theta1_hist, label=r'$\theta_1$')
plt.plot(iters, theta0_hist, label=r'$\theta_0$')

plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Parameter values over iterations")
plt.legend()
plt.savefig("fop")
plt.show()