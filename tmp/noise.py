import random


def add_variance(value: float) -> tuple[float, float]:
    percent = random.uniform(0.15, 0.30)
    direction = random.choice((-1, 1))
    varied_value = value * (1 + (direction * percent))
    return varied_value, direction * percent


def main() -> None:
    poly_coefficients = [7, 5, -3, -2]
    results = []
    for coeff in poly_coefficients:
        results.append(add_variance(coeff))

    for i, result in enumerate(results):
        print(f"Original value: {poly_coefficients[i]}, Value+variance: {result[0]}, Percentage: {result[1]*100:.2f}")

if __name__ == "__main__":
    main()
