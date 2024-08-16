from functools import reduce
from operator import mul

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def product(iterable):
    return reduce(mul, iterable, 1)


def matrix_zeros(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]


def vector_zeros(length):
    return [0 for _ in range(length)]


def lu_solve(A, b):
    # LU decomposition
    N = len(A)
    L = [[0.0] * N for _ in range(N)]
    U = [[0.0] * N for _ in range(N)]

    for i in range(N):
        L[i][i] = 1.0

        for j in range(i + 1):
            s1 = 0.0
            for k in range(j):
                s1 += U[k][i] * L[j][k]
            U[j][i] = A[j][i] - s1

        for j in range(i, N):
            s2 = 0.0
            for k in range(i):
                s2 += U[k][i] * L[j][k]
            L[j][i] = (A[j][i] - s2) / U[i][i]

    # Solving system
    y = [0.0 for _ in range(N)]
    x = [0.0 for _ in range(N)]

    for i in range(N):
        sum_Ly = 0.0
        for j in range(i):
            sum_Ly += L[i][j] * y[j]
        y[i] = b[i] - sum_Ly

    for i in range(N - 1, -1, -1):
        sum_Ux = 0.0
        for j in range(i + 1, N):
            sum_Ux += U[i][j] * x[j]
        x[i] = (y[i] - sum_Ux) / U[i][i]

    return x


def lagrange_interpolation(points):
    def f(x):
        result = 0
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            base = 1
            for j in range(n):
                if i == j:
                    continue
                else:
                    xj, yj = points[j]
                    base *= (float(x) - float(xj)) / float(float(xi) - float(xj))
            result += float(yi) * base
        return result

    return f


def spline_interpolation(points):
    def calculate_params():
        n = len(points)

        # we have n points => n-1 intervals => 4*(n-1) equations
        # => 4*(n-1) x 4*(n-1) matrix A, and 4*(n-1) - element vectors x and b.
        # x = [a0, b0, c0, d0, a1, ...., an-1, bn-1, cn-1, dn-1], len(x) = 4*(n-1), where n = len(interpolation_data)

        A = matrix_zeros(4 * (n - 1), 4 * (n - 1))
        b = vector_zeros(4 * (n - 1))

        # step 1: Si(xj) = f(xj)
        # n intervals => n-1 equations

        for i in range(n - 1):
            x, y = points[i]
            row = vector_zeros(4 * (n - 1))
            row[4 * i + 3] = 1
            A[4 * i + 3] = row
            b[4 * i + 3] = (float(y))

        # step 2: Sj(Xj+1) = f(Xj+1)
        # n intervals => n-1 equations
        # total : 2n-2 equations

        for i in range(n - 1):
            x1, y1 = points[i + 1]
            x0, y0 = points[i]
            h = float(x1) - float(x0)
            row = vector_zeros(4 * (n - 1))
            row[4 * i] = h ** 3
            row[4 * i + 1] = h ** 2
            row[4 * i + 2] = h ** 1
            row[4 * i + 3] = 1
            A[4 * i + 2] = row
            b[4 * i + 2] = float(y1)

        # step 3: for inner points, Sj-1'(xj) = Sj'(xj)
        # n points => n-2 inner points => n-2 equations
        # total : 3n-4 equations

        for i in range(n - 2):
            x1, y1 = points[i + 1]
            x0, y0 = points[i]
            h = float(x1) - float(x0)
            row = vector_zeros(4 * (n - 1))
            row[4 * i] = 3 * (h ** 2)
            row[4 * i + 1] = 2 * h
            row[4 * i + 2] = 1
            row[4 * (i + 1) + 2] = -1
            A[4 * i] = row
            b[4 * i] = float(0)

        # step 4: for inner points, Sj-1''(xj) = Sj''(xj)
        # n points => n-2 inner points => n-2 equations
        # total : 4n-6 equations

        for i in range(n - 2):
            x1, y1 = points[i + 1]
            x0, y0 = points[i]
            h = float(x1) - float(x0)
            row = vector_zeros(4 * (n - 1))
            row[4 * i] = 6 * h
            row[4 * i + 1] = 2
            row[4 * (i + 1) + 1] = -2
            A[4 * (i + 1) + 1] = row
            b[4 * (i + 1) + 1] = float(0)

        # step 5: on edges: S0''(x0) = 0 and Sn-1''(xn-1) = 0
        # 2 equations
        # total : 4n-4 equations

        # first point
        row = vector_zeros(4 * (n - 1))
        row[1] = 2
        A[1] = row
        b[1] = float(0)

        # last point
        row = vector_zeros(4 * (n - 1))
        x1, y1 = points[-1]
        x0, y0 = points[-2]
        h = float(x1) - float(x0)
        row[1] = 2
        row[-4] = 6 * h
        A[-4] = row
        b[-4] = float(0)

        result = lu_solve(A, b)
        return result

    params = calculate_params()

    def f(x):
        param_array = []
        row = []
        for param in params:
            row.append(param)
            if len(row) == 4:
                param_array.append(row.copy())
                row.clear()

        for i in range(1, len(points)):
            xi, yi = points[i - 1]
            xj, yj = points[i]
            if float(xi) <= x <= float(xj):
                a, b, c, d = param_array[i - 1]
                h = x - float(xi)
                return a * (h ** 3) + b * (h ** 2) + c * h + d

        return -1

    return f


def plot_interpolation(x_values, interp_func, x_all, y_all, x, y, title, file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, [interp_func(xi) for xi in x_values], label=title)
    plt.plot(x_all, y_all, label='Actual terrain', color='red')
    plt.scatter(x, y, color='green', s=50)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()
    print(f"File created successfully: {file_name}")


def plot_interpolations():
    num_points_list = [5, 10, 25, 50]
    csv_files = {
        "MountEverest.csv": "everest",
        "SpacerniakGdansk.csv": "spacerniak",
        "WielkiKanionKolorado.csv": "grand_canyon"
    }

    csv_dir = "csv"

    for file_name, folder_name in csv_files.items():
        file_path = os.path.join(csv_dir, file_name)
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, delimiter=",")
            x_all = data['Dystans (m)'].tolist()
            y_all = data['Wysokość (m)'].tolist()
            all_points = list(zip(x_all, y_all))

            for num_points in num_points_list:
                indices = np.linspace(0, len(x_all) - 1, num_points, dtype=int)
                points = [all_points[i] for i in indices]
                x = [point[0] for point in points]
                y = [point[1] for point in points]

                lagrange_func = lagrange_interpolation(points)
                spline_func = spline_interpolation(points)

                x_values = np.linspace(min(x_all), max(x_all), num=1000)

                lagrange_dir = f"plots/{folder_name}/lagrange"
                spline_dir = f"plots/{folder_name}/spline"
                os.makedirs(lagrange_dir, exist_ok=True)
                os.makedirs(spline_dir, exist_ok=True)

                lagrange_file_name = f"{lagrange_dir}/Lagrange-{file_name}-{num_points}-points.png"
                plot_interpolation(x_values, lagrange_func, x_all, y_all, x, y,
                                   f"Lagrange - {file_name} - {num_points} points",
                                   lagrange_file_name)

                spline_file_name = f"{spline_dir}/Spline-{file_name}-{num_points}-points.png"
                plot_interpolation(x_values, spline_func, x_all, y_all, x, y,
                                   f"Spline - {file_name} - {num_points} points",
                                   spline_file_name)
        else:
            print(f"File not found: {file_path}")


plot_interpolations()
