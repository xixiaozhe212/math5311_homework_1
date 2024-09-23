import numpy as np
import matplotlib.pyplot as plt

def heat_conduction_implicit(L, N, T, f):
    dx = L / N
    x = np.linspace(0, L, N + 1)
    M = int(np.ceil(T / (0.5 * dx * dx)))
    dt = T / M
    t = np.linspace(0, T, M + 1)
    u = np.zeros((N + 1, M + 1))
    # 初始条件
    for i in range(N + 1):
        u[i, 0] = np.sin(x[i])
    # 边界条件
    for j in range(M + 1):
        u[0, j] = 0
        u[N, j] = 0
    # 构建系数矩阵
    a = -dt / dx**2
    b = 1 + 2 * dt / dx**2
    c = -dt / dx**2
    A = np.diag([b] * (N - 1)) + np.diag([a] * (N - 2), k=-1) + np.diag([c] * (N - 2), k=1)
    # 迭代求解
    for j in range(1, M + 1):
        # 构建右侧向量
        r = u[1:N, j - 1] + dt * np.array([f(x[i], t[j - 1]) for i in range(1, N)])
        u[1:N, j] = np.linalg.solve(A, r)
    return x, t, u

def f_function(x, t):
    return np.sin(x) * (np.cos(t) - np.sin(t))

def exact_solution(x, t):
    return np.sin(x) * np.cos(t)

L = np.pi
T = 1.0

dx_values = []
max_errors = []

for N in [20, 40, 80, 160]:
    x, t, u = heat_conduction_implicit(L, N, T, f_function)
    # Calculate exact solution for each x at the last time step
    u_exact = exact_solution(x, t[-1])
    # Calculate the absolute difference between numerical and exact solutions
    diff = np.abs(u[:, -1] - u_exact)
    # Find the maximum error
    error = np.max(diff)
    dx_values.append(L / N)
    max_errors.append(error)

plt.plot(dx_values, max_errors, label="Maximum Error vs dx")
plt.xlabel("dx")
plt.ylabel("Maximum Error")
plt.legend()
plt.show()