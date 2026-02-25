import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random


def my_time(func, *args):
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    return end - start


def py_mean(lst):
    return sum(lst) / len(lst)


def py_multiply(lst, factor):
    return [x * factor for x in lst]


sizes = [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]

results = []

SCAL = 2

for size in sizes:

    python_list = [random.randint(1, 100) for _ in range(size)]
    numpy_array = np.array(python_list)

    t_py_sum = my_time(sum, python_list)

    t_py_mean = my_time(py_mean, python_list)

    t_py_mul = my_time(py_multiply, python_list, SCAL)

    t_np_sum = my_time(np.sum, numpy_array)

    t_np_mean = my_time(np.mean, numpy_array)

    t_np_mul = my_time(np.multiply, numpy_array, SCAL)

    results.append(
        {
            "size": size,
            "py_sum_time": t_py_sum,
            "np_sum_time": t_np_sum,
            "py_mean_time": t_py_mean,
            "np_mean_time": t_np_mean,
            "py_mul_time": t_py_mul,
            "np_mul_time": t_np_mul,
        }
    )


print("-" * 200)
header = (
    f"{'Size':>10} | {'Py sum':>12} | {'NP sum':>12} | {'Faster sum':>12} | "
    f"{'Py mean':>12} | {'NP mean':>12} | {'Faster mean':>12} | "
    f"{'Py mul':>12} | {'NP mul':>12} | {'Faster mul':>12}"
)
print(header)
print("-" * 200)

for r in results:
    faster_sum = "NumPy" if r["np_sum_time"] < r["py_sum_time"] else "Python"
    faster_mean = "NumPy" if r["np_mean_time"] < r["py_mean_time"] else "Python"
    faster_mul = "NumPy" if r["np_mul_time"] < r["py_mul_time"] else "Python"

    line = (
        f"{r['size']:>10} | {r['py_sum_time']:>12.6f} | {r['np_sum_time']:>12.6f} | {faster_sum:>12} | "
        f"{r['py_mean_time']:>12.6f} | {r['np_mean_time']:>12.6f} | {faster_mean:>12} | "
        f"{r['py_mul_time']:>12.6f} | {r['np_mul_time']:>12.6f} | {faster_mul:>12}"
    )
    print(line)

print("-" * 200)
