using Plots

function heat_conduction_explicit(f, alpha, nx, nt, L, T)
    dx = L / (nx - 1)
    dt = T / (nt - 1)
    lambda = alpha * dt / dx^2
    u = zeros(nx, nt)
    x = range(0, stop=L, length=nx)
    t = range(0, stop=T, length=nt)

    # 初始条件
    for i = 1:nx
        u[i, 1] = sin(pi * x[i])
    end

    # 边界条件
    for j = 1:nt
        u[1, j] = 0
        u[nx, j] = 0
    end

    # 构建系数矩阵 A
    A = zeros(nx - 2, nx - 2)
    for i = 1:nx - 2
        A[i, i] = 1 + 2 * lambda
        if i > 1
            A[i, i - 1] = -lambda
        end
        if i < nx - 2
            A[i, i + 1] = -lambda
        end
    end

    # 迭代求解
    for j = 2:nt
        b = zeros(nx - 2)
        for i = 2:nx - 1
            b[i - 1] = u[i, j - 1] + dt * Fx(x[i], t[j - 1])
        end
        u[2:nx - 1, j] = A \ b
    end

    return x, t, u
end

function heat_conduction_implicit(Fx, alpha, nx, nt, L, T)
    dx = L / (nx - 1)
    dt = T / (nt - 1)
    lambda = alpha * dt / dx^2
    u = zeros(nx, nt)
    x = range(0, stop=L, length=nx)
    t = range(0, stop=T, length=nt)

    # 初始条件
    for i = 1:nx
        u[i, 1] = sin(pi * x[i])
    end

    # 边界条件
    for j = 1:nt
        u[1, j] = 0
        u[nx, j] = 0
    end

    # 构建系数矩阵 A
    A = zeros(nx - 2, nx - 2)
    for i = 1:nx - 2
        A[i, i] = 1 + 2 * lambda
        if i > 1
            A[i, i - 1] = -lambda
        end
        if i < nx - 2
            A[i, i + 1] = -lambda
        end
    end

    # 迭代求解
    for j = 2:nt
        b = zeros(nx - 2)
        for i = 2:nx - 1
            b[i - 1] = u[i, j - 1] + dt * Fx(x[i], t[j - 1])
        end
        u[2:nx - 1, j] = A \ b
    end

    return x, t, u
end

function Fx_function(x, t)
    return x * t
end

function exact_solution(x, t)
    return exp(-pi^2 * t) * sin(pi * x)
end

alpha = 0.1
L = 1.0
T = 1.0

dx_values = []
errors = []

for nx in [11, 21, 51, 101]
    nt = 1001
    x, t, u = heat_conduction_implicit(Fx_function, alpha, nx, nt, L, T)
    u_exact = exact_solution.(x, t[:, end])
    error = norm(u[:, end] - u_exact)
    push!(dx_values, L / (nx - 1))
    push!(errors, error)
end

plot(dx_values, errors, xlabel="dx", ylabel="Error", label="Error vs dx")