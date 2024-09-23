using Plots

function heat_conduction_explicit(L, N, T, f, alpha)
    # 求 dx
    dx = L / N
    # 创建一个从 0 到 L 的一维数组，表示空间位置，长度为 N+1，即有 N+1 个空间网格点的位置信息
    x = range(0, stop=L, length=N+1)

    # 求 dt
    function ceil(B)
        integer_part = Int(floor(B))
        fractional_part = B - integer_part
        if fractional_part > 0
            return integer_part + 1
        else
            return integer_part
        end
    end
    M = ceil(T/(0.5 * dx * dx))
    dt = T / M
    t = range(0, stop=T, length=M+1)
    
    # 创建一个大小为 N+1（空间网格点数）行和 M+1（时间步数）列的矩阵，用于存储在不同空间位置和时间点上的温度值，初始时所有元素都为 0。
    u = zeros(N+1, M+1)

    # 初始条件
    for i = 1:N+1
        u[i, 1] = sin(x[i])
    end

    # 边界条件
    for j = 1:M+1
        u[1, j] = 0
        u[N+1, j] = 0
    end

    # 迭代求解，这里显式地使用 x 和 t 的值
    for j = 2:M+1
        for i = 2:N
            u[i, j] = u[i, j - 1] + dt / dx^2 * (u[i + 1, j - 1] - 2 * u[i, j - 1] + u[i - 1, j - 1]) + dt * f(x[i], t[j - 1])
        end
    end

    return x, t, u
end

function f_function(x, t)
    return sin(x) * (cos(t) - sin(t))
end

function exact_solution(x, t)
    return sin(x) * cos(t)
end

L = pi
T = 1.0

dx_values = []
max_errors = []

for N in [20, 40, 80, 160]
    x, t, u = heat_conduction_explicit(L, N, T, f_function, alpha)
    u_exact = exact_solution.(x, t[:, end])
    error = maximum(abs.(u[:, end].- u_exact))
    push!(dx_values, L / N)
    push!(max_errors, error)
end

plot(dx_values, max_errors, xlabel="dx", ylabel="Maximum Error", label="Maximum Error vs dx")