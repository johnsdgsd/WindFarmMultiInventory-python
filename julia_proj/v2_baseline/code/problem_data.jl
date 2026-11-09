# problem_data.jl — V2 连续基线（Python 等级模型 scope=none）的问题数据。
#
# 与 Python 版 code/problem_data.py / code/config.py 逐项对应：
#   · G 个运行等级 0..G-1，等级 G 为失效态；
#   · 月增量 D ~ Poisson(λ0·exp(β·(z−z̄)))；P 为 G×(G+1) 上三角，
#     p_digits 位四舍五入（对角吸收舍入误差）；
#   · 场景池每条 (z, Θ)，Θ 为 (N_R,G,G+1)，逐行 Dirichlet(sP)；
#     池全阶段共享、每阶段独立等概率抽样（由 model.jl 的 parameterize 实现）；
#   · 初始 n0[r,i] = N_r·init_mix[i]（连续，不取整）。
#
# 参数唯一事实源是 config/problem.toml（默认指向 Python 项目那份，只读）。
# 本模块只负责读取、校验、派生，不在代码里维护第二套问题参数默认值。
#
# 场景池的唯一事实源是 Python 导出的 data/pool_seed<seed>.txt：
# NumPy default_rng 的流无法在 Julia 逐位复现，默认禁止 Julia 自行采样；
# 只有显式 allow_julia_pool=true 才回退到 Julia RNG（并打印警告，结果不可与
# Python 逐场景对照）。导出脚本：scripts/export_pool.py。

using Random, Distributions, TOML, Printf, Statistics, LinearAlgebra

const COV_KEYS = ("mu_T", "sigma_T", "lambda", "alpha_L", "beta_L")

# v2_baseline/code/problem_data.jl -> v2_baseline
const PROJECT_ROOT = dirname(@__DIR__)

"""默认问题配置：优先环境变量 V2_PROBLEM_TOML，其次 Python 项目的
config/problem.toml（本仓库只读，不复制、不改写）。"""
function default_problem_toml()
    env = get(ENV, "V2_PROBLEM_TOML", "")
    if !isempty(env)
        p = abspath(env)
        isfile(p) || error("V2_PROBLEM_TOML 指向的文件不存在：$p")
        return p
    end
    p = normpath(joinpath(PROJECT_ROOT, "..", "..", "config", "problem.toml"))
    isfile(p) || error(
        "找不到默认配置 $(p)。请显式传 config_path，或设置环境变量 V2_PROBLEM_TOML。")
    return p
end

"""默认场景池文件：v2_baseline/data/pool_seed<seed>.txt（由 Python 导出）。"""
default_pool_path(seed::Int) =
    joinpath(PROJECT_ROOT, "data", "pool_seed$(seed).txt")

# ---------------------------------------------------------------- 配置读取
# 缺键直接报错：默认值只在 config/problem.toml 一处，代码里不兜底。
function _req(d::AbstractDict, key::String, where::String)
    haskey(d, key) || error(
        "配置 $where 缺少必填项 \"$key\"。默认值请写在 config/problem.toml，不要在代码里兜底。")
    return d[key]
end

# 仅用于旧配置的兼容项（I0 缺省全 0、n0_mode 缺省 continuous），与 Python 的
# config.py::_opt 一致；决定模型结构的参数一律走 _req。
_opt(d::AbstractDict, key::String, default, ::String) = haskey(d, key) ? d[key] : default

function _section(d::AbstractDict, name::String, where::String)
    sec = _req(d, name, where)
    sec isa AbstractDict || error("配置 $where 的 [$name] 必须是表（节）")
    return sec
end

"""读取 problem.toml，返回原始参数（不做物理校验；校验在 DefineProblem）。"""
function load_config(path::AbstractString)
    p = abspath(path)
    isfile(p) || error("配置文件不存在：$p")
    raw = TOML.parsefile(p)
    where = basename(p)
    dim = _section(raw, "dimension", where)
    fleet = _section(raw, "fleet", where)
    cost = _section(raw, "cost", where)
    deg = _section(raw, "degradation", where)
    cov_raw = _section(raw, "covariate", where)
    scen = _section(raw, "scenario", where)

    priority = nothing
    if haskey(raw, "priority")
        prio_raw = raw["priority"]
        prio_raw isa AbstractDict || error("配置 $where 的 [priority] 必须是表")
        priority = Dict{Int,Vector{Int}}()
        for (k, v) in prio_raw
            priority[parse(Int, string(k))] = [Int(x) for x in v]
        end
    end

    return (;
        T = Int(_req(dim, "T", "$where/dimension")),
        G = Int(_req(dim, "G", "$where/dimension")),
        N_R = Int(_req(dim, "N_R", "$where/dimension")),
        L = Int(_req(dim, "L", "$where/dimension")),
        N = [Int(x) for x in _req(fleet, "N", "$where/fleet")],
        Qmax = [Int(x) for x in _req(fleet, "Qmax", "$where/fleet")],
        Imax = [Int(x) for x in _req(fleet, "Imax", "$where/fleet")],
        fleet_tol = Float64(_req(fleet, "fleet_tol", "$where/fleet")),
        state_round_digits = Int(_req(fleet, "state_round_digits", "$where/fleet")),
        state_repair_tol = Float64(_req(fleet, "state_repair_tol", "$where/fleet")),
        init_mix = [Float64(x) for x in _req(fleet, "init_mix", "$where/fleet")],
        n0_mode = string(_opt(fleet, "n0_mode", "continuous", "$where/fleet")),
        I0 = [Float64(x) for x in _opt(fleet, "I0", fill(0.0, Int(_req(dim, "N_R", "$where/dimension"))), "$where/fleet")],
        C = Float64(_req(cost, "C", "$where/cost")),
        h = Float64(_req(cost, "h", "$where/cost")),
        Cs = Float64(_req(cost, "Cs", "$where/cost")),
        atc = Float64(_req(cost, "atc", "$where/cost")),
        tc = Float64(_req(cost, "tc", "$where/cost")),
        lambda0 = Float64(_req(deg, "lambda0", "$where/degradation")),
        dirichlet_s = Float64(_req(deg, "dirichlet_s", "$where/degradation")),
        p_digits = Int(_req(deg, "p_digits", "$where/degradation")),
        beta_cov = [Float64(x) for x in _req(deg, "beta_cov", "$where/degradation")],
        z_bar = [Float64(x) for x in _req(deg, "z_bar", "$where/degradation")],
        cov = (mu_T = Float64(_req(cov_raw, "mu_T", "$where/covariate")),
               sigma_T = Float64(_req(cov_raw, "sigma_T", "$where/covariate")),
               lambda = Float64(_req(cov_raw, "lambda", "$where/covariate")),
               alpha_L = Float64(_req(cov_raw, "alpha_L", "$where/covariate")),
               beta_L = Float64(_req(cov_raw, "beta_L", "$where/covariate"))),
        pool_size = Int(_req(scen, "pool_size", "$where/scenario")),
        seed = Int(_req(scen, "seed", "$where/scenario")),
        priority = priority,
        path = p,
    )
end

# ---------------------------------------------------------------- 转移矩阵
"""Poisson 质量函数 p_k = e^{-λ}λ^k/k!。"""
poisson_pmf(lam::Float64, k::Int) = exp(-lam) * lam^k / factorial(k)

"""G×(G+1) 上三角转移矩阵，语义与 Python transition_matrix 逐位一致：

精确：P[i,j]=p_{j−i} (j<G)，P[i,G]=1−Σ_{k=0}^{G−1−i}p_k，j<i 为 0。
若 p_digits 非空：非对角分量（含失效列）四舍五入；对角吸收舍入误差；
若非对角和 >1 先归一化。最后裁掉负零并逐行归一化。

Julia 1-based：行 i=1..G 对应 Python i=0..G-1；列 j=1..G+1 对应 Python 0..G。"""
function transition_matrix(lam::Real, G::Int; p_digits::Union{Nothing,Int} = 3)
    lam = Float64(lam)
    lam >= 0 || error("lambda 必须非负，实际 $lam")
    p = [poisson_pmf(lam, k) for k in 0:G]     # p[1] = p_0
    P = zeros(Float64, G, G + 1)
    for i in 1:G                              # Python i0 = i-1
        off = Float64[]
        for j in (i + 1):(G + 1)              # Python j0 = j-1
            if j <= G
                push!(off, p[j - i + 1])      # p_{j0-i0} = p_{j-i}
            else
                # tail: 1 - Σ_{k=0}^{G-i0-1} p_k = 1 - Σ_{k=0}^{G-i} p_k
                push!(off, 1.0 - sum(p[1:(G - i + 1)]))
            end
        end
        if p_digits !== nothing
            off = round.(off; digits = p_digits)
            s = sum(off)
            if s > 1.0 + 1e-12
                off ./= s
            end
        end
        P[i, i + 1:G + 1] = off
        P[i, i] = 1.0 - sum(off)
    end
    P = max.(P, 0.0)
    for i in 1:G
        rs = sum(@view P[i, :])
        rs > 0 || error("转移矩阵出现空行（i=$(i)）")
        @views P[i, :] ./= rs
    end
    return P
end

"""T 期内从 init_mix 出发的累计失效概率（P 的失效列）。"""
function cumulative_failure_prob(P::AbstractMatrix, T::Int, init_mix::AbstractVector)
    G = size(P, 1)
    Q = P[:, 1:G]
    f = P[:, G + 1]
    v = collect(Float64, init_mix)
    acc = 0.0
    for _ in 1:T
        acc += dot(v, f)
        v = vec(transpose(v) * Q)
    end
    return acc
end

# ---------------------------------------------------------------- PHM 因子
"""修正因子 f_r = exp(Σ_k β_k(z_k − z̄_k))；乘在 λ0 上。"""
_cov_factor(beta_cov::AbstractVector, z_bar::AbstractVector, z_r::AbstractVector) =
    exp(dot(beta_cov, z_r .- z_bar))

# ---------------------------------------------------------------- 优先级/路线
"""默认调拨优先级表。N_R=5 时沿用 MutiInventory 的实测表（与 Python 一致）。"""
function default_priority(N_R::Int)
    if N_R == 5
        return Dict(1 => [1, 2, 3, 4, 5], 2 => [2, 1, 3, 4, 5],
                    3 => [3, 1, 2, 4, 5], 4 => [4, 1, 2, 3, 5],
                    5 => [5, 1, 2, 3, 4])
    end
    return Dict(r => vcat(r, [m for m in 1:N_R if m != r]) for r in 1:N_R)
end

"""路线差异化被动调拨成本：m 在 r 优先级表中的位置 pos → (pos−1)·tc。"""
function make_tc_route(priority::Dict{Int,Vector{Int}}, tc::Float64, N_R::Int)
    route = zeros(Float64, N_R, N_R)
    for r in 1:N_R
        for (pos, m) in enumerate(priority[r])
            route[r, m] = (pos - 1) * tc
        end
    end
    return route
end

# ---------------------------------------------------------------- 场景池
"""Julia RNG 回退采样（默认不用）。分布与 Python 相同，但随机流不同，
因此场景不可能与 Python 逐位一致；只在 allow_julia_pool=true 时调用。"""
function sample_pool(lambda0::Float64, beta_cov::Vector{Float64},
                     z_bar::Vector{Float64}, cov::NamedTuple,
                     N_R::Int, G::Int, dirichlet_s::Float64,
                     p_digits::Int, pool_size::Int, seed::Int)
    rng = MersenneTwister(seed)
    pool = Vector{Tuple{Matrix{Float64},Array{Float64,3}}}(undef, pool_size)
    for s in 1:pool_size
        z = Matrix{Float64}(undef, 3, N_R)
        z[1, :] = rand(rng, Normal(cov.mu_T, cov.sigma_T), N_R)
        z[2, :] = Float64.(rand(rng, Poisson(cov.lambda), N_R))
        z[3, :] = 100.0 .* rand(rng, Beta(cov.alpha_L, cov.beta_L), N_R)
        theta = zeros(Float64, N_R, G, G + 1)
        for r in 1:N_R
            f = _cov_factor(beta_cov, z_bar, @view z[:, r])
            P = transition_matrix(lambda0 * f, G; p_digits = p_digits)
            for i in 1:G
                cols = [j for j in i:(G + 1) if P[i, j] > 0.0]
                isempty(cols) && error("P 第 $i 行无正支撑")
                vals = if length(cols) == 1
                    [1.0]
                else
                    rand(rng, Dirichlet([dirichlet_s * P[i, j] for j in cols]))
                end
                theta[r, i, cols] = vals
            end
        end
        for r in 1:N_R, i in 1:G
            abs(sum(@view theta[r, i, :]) - 1.0) <= 1e-12 ||
                error("Θ 行和偏离 1（r=$(r),i=$(i)）")
            for j in 1:(i - 1)
                theta[r, i, j] == 0.0 ||
                    error("Θ 下三角必须为 0（r=$(r),i=$(i),j=$(j)）")
            end
        end
        pool[s] = (z, theta)
    end
    return pool
end

"""读取 Python 导出的池文件。

文件格式见 scripts/export_pool.py：首行 `pool_size N_R G T seed`，随后每行
一条场景，前 3·N_R 个值为 z（C order），后 N_R·G·(G+1) 个值为 Θ（C order）。
返回 (pool, meta)。`pool_size` 给定时取前若干条（只用于冒烟；不改变文件）。"""
function load_pool_file(path::AbstractString, N_R::Int, G::Int;
                        pool_size::Union{Nothing,Int} = nothing,
                        seed::Union{Nothing,Int} = nothing)
    p = abspath(path)
    isfile(p) || error("场景池文件不存在：$p")
    lines = readlines(p)
    isempty(lines) && error("场景池文件为空：$p")
    hdr = split(strip(lines[1]))
    length(hdr) >= 5 || error("池文件头应为 `pool_size N_R G T seed`，实际：$(lines[1])")
    file_n, file_NR, file_G, file_T, file_seed = parse.(Int, hdr[1:5])
    file_NR == N_R || error("池文件 N_R=$file_NR ≠ 配置 N_R=$N_R")
    file_G == G || error("池文件 G=$file_G ≠ 配置 G=$G")
    if seed !== nothing && seed != file_seed
        error("请求 seed=$seed 与池文件 seed=$file_seed 不一致；池文件是场景唯一事实源。"
              * "请先用 scripts/export_pool.py 导出该 seed 的池。")
    end
    n_use = pool_size === nothing ? file_n : pool_size
    n_use >= 2 || error("pool_size 至少为 2，实际 $n_use")
    n_use <= file_n || error("请求 pool_size=$n_use 超过池文件条数 $file_n")
    n_use < file_n && @info "使用池文件前 $n_use/$file_n 条场景（冒烟子集；权重仍等概率）"

    n_z = 3 * N_R
    n_th = N_R * G * (G + 1)
    pool = Vector{Tuple{Matrix{Float64},Array{Float64,3}}}(undef, n_use)
    for s in 1:n_use
        1 + s <= length(lines) || error("池文件行数不足：需要 $(1 + s) 行，实际 $(length(lines))")
        vals = parse.(Float64, split(strip(lines[1 + s])))
        length(vals) == n_z + n_th || error(
            "场景 $s 的数值个数 $(length(vals)) ≠ $(n_z + n_th)")
        z = Matrix{Float64}(undef, 3, N_R)
        k = 0
        # Python 的 z.ravel(order="C")：r 变化最快
        for i in 1:3, r in 1:N_R
            k += 1
            z[i, r] = vals[k]
        end
        theta = zeros(Float64, N_R, G, G + 1)
        # Python 的 Θ.ravel(order="C")：j 最快、i 次之、r 最慢
        for r in 1:N_R, i in 1:G, j in 1:(G + 1)
            k += 1
            theta[r, i, j] = vals[k]
        end
        pool[s] = (z, theta)
    end
    return pool, (file_pool_size = file_n, N_R = file_NR, G = file_G,
                  T = file_T, seed = file_seed, path = p)
end

# ---------------------------------------------------------------- 数据结构
"""V2 连续基线问题数据。字段顺序/含义对应 Python ProblemData（scope=none）。"""
struct ProblemData
    # 维度
    T::Int
    G::Int
    N_R::Int
    L::Int
    # 机队与初值
    N::Vector{Int}
    n0::Matrix{Float64}          # (N_R,G)，continuous：N_r·init_mix
    I0::Vector{Float64}
    o0::Matrix{Float64}          # (L-1,N_R)
    init_mix::Vector{Float64}
    # 成本
    C::Float64
    h::Float64
    Cs::Float64
    atc::Float64
    tc::Float64
    # 容量/数值
    Qmax::Vector{Int}
    Imax::Vector{Int}
    fleet_tol::Float64
    state_round_digits::Int      # 仅 Python 状态流转用；本模型不实现自定义仿真
    state_repair_tol::Float64    # 同上
    # 被动调拨
    priority::Dict{Int,Vector{Int}}
    tc_route::Matrix{Float64}
    # 退化/转移
    lambda0::Float64
    dirichlet_s::Float64
    p_digits::Int
    beta_cov::Vector{Float64}
    z_bar::Vector{Float64}
    cov::NamedTuple{(:mu_T, :sigma_T, :lambda, :alpha_L, :beta_L),NTuple{5,Float64}}
    P_ref::Matrix{Float64}
    # 场景池
    pool::Vector{Tuple{Matrix{Float64},Array{Float64,3}}}
    seed::Int
    # 来源留痕
    config_path::String
    pool_source::String
end

state_dim(prob::ProblemData) = prob.N_R + (prob.L - 1) * prob.N_R +
                               prob.N_R * prob.G + prob.N_R
total_N(prob::ProblemData) = sum(prob.N)

"""
    DefineProblem(; config_path, pool_path, T, pool_size, seed, allow_julia_pool)

构造 V2 连续基线问题数据。参数全部来自 config_path（默认 Python 项目的
config/problem.toml）；T/pool_size/seed 只用于冒烟覆盖。场景默认从
data/pool_seed<seed>.txt 读取（Python 导出的同一批场景）；找不到时默认报错，
只有 allow_julia_pool=true 才用 Julia RNG 回退并告警。
"""
function DefineProblem(;
        config_path::AbstractString = default_problem_toml(),
        pool_path::Union{Nothing,AbstractString} = nothing,
        T::Union{Nothing,Int} = nothing,
        pool_size::Union{Nothing,Int} = nothing,
        seed::Union{Nothing,Int} = nothing,
        allow_julia_pool::Bool = false,
    )
    cfg = load_config(config_path)
    T_eff = T === nothing ? cfg.T : Int(T)
    seed_eff = seed === nothing ? cfg.seed : Int(seed)
    where = basename(cfg.path)

    # ---- 维度校验（报错带具体数值） ----
    T_eff > 0 || error("[dimension] T 必须为正，实际 $T_eff")
    cfg.G >= 2 || error("[dimension] G（运行等级数）必须 ≥ 2，实际 $(cfg.G)")
    cfg.N_R > 0 || error("[dimension] N_R 必须为正，实际 $(cfg.N_R)")
    cfg.L >= 2 || error("[dimension] 当前实现要求 L ≥ 2（在途管 L−1 ≥ 1），实际 $(cfg.L)")

    for name in (:N, :Qmax, :Imax)
        v = getfield(cfg, name)
        length(v) == cfg.N_R || error(
            "[fleet] $name 长度 $(length(v)) ≠ N_R=$(cfg.N_R)；改 N_R 时必须同步改写列表")
        any(x -> x < 0, v) && error("[fleet] $name 不得含负值：$v")
    end
    any(==(0), cfg.Qmax) && error("[fleet] Qmax 不得含 0（仓库无法订货）：$(cfg.Qmax)")
    0.0 <= cfg.fleet_tol <= 1e-3 || error(
        "[fleet] fleet_tol 必须在 [0, 1e-3] 台，实际 $(cfg.fleet_tol)")
    0 <= cfg.state_round_digits <= 12 || error(
        "[fleet] state_round_digits 必须在 0..12，实际 $(cfg.state_round_digits)")
    cfg.fleet_tol <= cfg.state_repair_tol <= 1e-2 || error(
        "[fleet] state_repair_tol 必须在 [fleet_tol, 1e-2] 台，实际 $(cfg.state_repair_tol)")
    length(cfg.init_mix) == cfg.G || error(
        "[fleet] init_mix 长度 $(length(cfg.init_mix)) ≠ G=$(cfg.G)")
    any(x -> x < 0, cfg.init_mix) && error("[fleet] init_mix 不得含负值：$(cfg.init_mix)")
    abs(sum(cfg.init_mix) - 1.0) <= 1e-8 || error(
        "[fleet] init_mix 必须和为 1，实际 $(sum(cfg.init_mix))")
    cfg.n0_mode == "continuous" || error(
        "[fleet] 本 V2 连续基线只实现 n0_mode=continuous，实际 $(cfg.n0_mode)")
    length(cfg.I0) == cfg.N_R || error(
        "[fleet] I0 长度 $(length(cfg.I0)) ≠ N_R=$(cfg.N_R)")
    any(x -> x < 0, cfg.I0) && error("[fleet] I0 不得含负值：$(cfg.I0)")
    bad = [r for r in 1:cfg.N_R if cfg.I0[r] > cfg.Imax[r]]
    isempty(bad) || error(
        "[fleet] I0 超过 Imax：仓库 $(bad)，I0=$(cfg.I0), Imax=$(cfg.Imax)")

    cfg.lambda0 > 0 || error("[degradation] lambda0 必须为正，实际 $(cfg.lambda0)")
    cfg.dirichlet_s > 0 || error(
        "[degradation] dirichlet_s 必须为正，实际 $(cfg.dirichlet_s)")
    0 <= cfg.p_digits <= 6 || error(
        "[degradation] p_digits 必须在 0..6，实际 $(cfg.p_digits)")
    (length(cfg.beta_cov) == 3 && length(cfg.z_bar) == 3) || error(
        "[degradation] PHM 要求 3 维协变量，实际 beta_cov=$(cfg.beta_cov)、z_bar=$(cfg.z_bar)")
    for k in ("sigma_T", "lambda", "alpha_L", "beta_L")
        getfield(cfg.cov, Symbol(k)) > 0 || error(
            "[covariate] $k 必须为正，实际 $(getfield(cfg.cov, Symbol(k)))")
    end

    pool_size_eff = pool_size === nothing ? cfg.pool_size : Int(pool_size)
    pool_size_eff >= 2 || error("[scenario] pool_size 至少为 2，实际 $pool_size_eff")

    # ---- 被动调拨优先级 ----
    priority = cfg.priority === nothing ? default_priority(cfg.N_R) : cfg.priority
    sort(collect(keys(priority))) == collect(1:cfg.N_R) || error(
        "[priority] 必须恰覆盖仓库 1..N_R（N_R=$(cfg.N_R)），实际键 $(sort(collect(keys(priority))))")
    for r in 1:cfg.N_R
        sort(priority[r]) == collect(1:cfg.N_R) || error(
            "[priority] 仓库 $r 的优先级表必须恰含 1..N_R，实际 $(priority[r])")
    end
    tc_route = make_tc_route(priority, cfg.tc, cfg.N_R)

    # ---- 初始状态（continuous） ----
    n0 = zeros(Float64, cfg.N_R, cfg.G)
    for r in 1:cfg.N_R, i in 1:cfg.G
        n0[r, i] = cfg.N[r] * cfg.init_mix[i]
    end
    for r in 1:cfg.N_R
        abs(sum(@view n0[r, :]) - cfg.N[r]) <= 1e-8 || error(
            "仓库 $r 初始等级分布和 $(sum(n0[r, :])) ≠ N=$(cfg.N[r])")
    end
    o0 = zeros(Float64, cfg.L - 1, cfg.N_R)

    P_ref = transition_matrix(cfg.lambda0, cfg.G; p_digits = cfg.p_digits)

    # ---- 场景池：默认必须来自 Python 导出的文件 ----
    pool_file = pool_path === nothing ? default_pool_path(seed_eff) : abspath(pool_path)
    if isfile(pool_file)
        if pool_size === nothing && cfg.pool_size != _pool_file_size(pool_file)
            error("配置 pool_size=$(cfg.pool_size) 与池文件条数 $(_pool_file_size(pool_file)) 不一致；"
                  * "请重新导出场景池，或显式传 pool_size 做冒烟子集。")
        end
        pool, meta = load_pool_file(pool_file, cfg.N_R, cfg.G;
                                    pool_size = pool_size_eff, seed = seed_eff)
        pool_source = meta.path
    elseif allow_julia_pool
        @warn "未找到 Python 导出的场景池 $(pool_file)；回退到 Julia RNG 采样。" *
              "场景与 Python 不同，结果不可逐场景对照。"
        pool = sample_pool(cfg.lambda0, cfg.beta_cov, cfg.z_bar, cfg.cov,
                           cfg.N_R, cfg.G, cfg.dirichlet_s, cfg.p_digits,
                           pool_size_eff, seed_eff)
        pool_source = "julia-rng:MersenneTwister($(seed_eff))"
    else
        error("找不到场景池文件 $(pool_file)。请先运行 "
              * "`python julia_proj/v2_baseline/scripts/export_pool.py` 导出 Python 的同一批场景；"
              * "若只做结构冒烟，可显式传 allow_julia_pool=true（场景将与 Python 不同）。")
    end

    return ProblemData(
        T_eff, cfg.G, cfg.N_R, cfg.L,
        cfg.N, n0, cfg.I0, o0, cfg.init_mix,
        cfg.C, cfg.h, cfg.Cs, cfg.atc, cfg.tc,
        cfg.Qmax, cfg.Imax, cfg.fleet_tol,
        cfg.state_round_digits, cfg.state_repair_tol,
        priority, tc_route,
        cfg.lambda0, cfg.dirichlet_s, cfg.p_digits,
        cfg.beta_cov, cfg.z_bar, cfg.cov, P_ref,
        pool, seed_eff, cfg.path, pool_source,
    )
end

function _pool_file_size(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && return 0
    hdr = split(strip(lines[1]))
    isempty(hdr) && return 0
    return tryparse(Int, hdr[1]) === nothing ? 0 : parse(Int, hdr[1])
end

# ---------------------------------------------------------------- 摘要/自检
"""打印参数摘要 + 场景池/矩匹配自检（构建规则 §1：print_summary 必带自检）。

注意：pool=4 之类的冒烟子集会让矩匹配偏差变大，这里只打印不报错；
行和/结构零是硬门槛，出错直接抛。"""
function print_summary(prob::ProblemData; io::IO = stdout)
    G, N_R = prob.G, prob.N_R
    println(io, "="^72)
    println(io, "问题数据摘要（V2 连续基线，等级模型 scope=none）")
    println(io, "="^72)
    println(io, "config = ", prob.config_path)
    println(io, "pool   = ", prob.pool_source)
    println(io, "T = $(prob.T)  G = $G  N_R = $N_R  L = $(prob.L)  ",
            "pool = $(length(prob.pool))  seed = $(prob.seed)  state_dim = $(state_dim(prob))")
    println(io, "机队 N = $(prob.N)  总数 = $(total_N(prob))  ",
            "init_mix = $(prob.init_mix)  n0_mode = continuous  I0 = $(prob.I0)")
    @printf(io, "fleet_tol = %g  state_round_digits = %d  state_repair_tol = %g\n",
            prob.fleet_tol, prob.state_round_digits, prob.state_repair_tol)
    println(io, "成本: C=$(prob.C)  h=$(prob.h)  Cs=$(prob.Cs)  atc=$(prob.atc)  tc=$(prob.tc)")
    println(io, "λ0=$(prob.lambda0)  dirichlet_s=$(prob.dirichlet_s)  p_digits=$(prob.p_digits)")
    println(io, "PHM: beta_cov=$(prob.beta_cov)  z_bar=$(prob.z_bar)")
    c = prob.cov
    println(io, "协变量: 油温 N($(c.mu_T), $(c.sigma_T)²)  ",
            "启停 Poisson($(c.lambda))  满负荷 100·Beta($(c.alpha_L), $(c.beta_L)) (%)")
    println(io, "P_ref（λ0 下，$(prob.p_digits) 位小数）:")
    for i in 1:G
        @printf(io, "  %s   sum=%.6f\n",
                join([@sprintf("%.6g", prob.P_ref[i, j]) for j in 1:(G + 1)], " "),
                sum(@view prob.P_ref[i, :]))
    end
    P_exact = transition_matrix(prob.lambda0, G; p_digits = nothing)
    @printf(io, "P_ref vs 精确矩阵: max|Δ| = %.3e\n",
            maximum(abs.(prob.P_ref .- P_exact)))

    pT = cumulative_failure_prob(prob.P_ref, prob.T, prob.init_mix)
    years = prob.T / 12.0
    @printf(io, "T=%d 期累计失效: %.4f%%  N=%d 时约 %.1f 台/年（按 T/12 折算）\n",
            prob.T, 100 * pT, total_N(prob), pT * total_N(prob) / years)

    # ---- 池自检 ----
    theta_bar = zeros(Float64, N_R, G, G + 1)
    for (_, th) in prob.pool
        theta_bar .+= th
    end
    theta_bar ./= length(prob.pool)
    Pz_bar = zeros(Float64, G, G + 1)
    for (z, _) in prob.pool, r in 1:N_R
        f = _cov_factor(prob.beta_cov, prob.z_bar, @view z[:, r])
        Pz_bar .+= transition_matrix(prob.lambda0 * f, G; p_digits = prob.p_digits)
    end
    Pz_bar ./= (length(prob.pool) * N_R)

    max_row = 0.0
    max_zero = 0.0
    for (_, th) in prob.pool, r in 1:N_R, i in 1:G
        max_row = max(max_row, abs(sum(@view th[r, i, :]) - 1.0))
        for j in 1:(i - 1)
            max_zero = max(max_zero, abs(th[r, i, j]))
        end
    end
    max_abs = 0.0
    max_rel = 0.0
    for r in 1:N_R, i in 1:G, j in 1:(G + 1)
        d = abs(theta_bar[r, i, j] - Pz_bar[i, j])
        max_abs = max(max_abs, d)
        if Pz_bar[i, j] >= 0.01
            max_rel = max(max_rel, d / Pz_bar[i, j])
        end
    end
    @printf(io, "池 Θ 行和 max|Δ|=%.2e  结构零 max|Δ|=%.2e\n", max_row, max_zero)
    @printf(io, "E[Θ] vs E_z[P(z)]: max|Δ|=%.4f  P≥0.01 处 max 相对误差=%.2f%%\n",
            max_abs, 100 * max_rel)
    max_row <= 1e-12 || error("池 Θ 行和自检失败：max|Δ|=$max_row")
    max_zero == 0.0 || error("池 Θ 结构零自检失败：max|Δ|=$max_zero")

    # ---- 协变量因子中心化（只打印） ----
    rng_th = MersenneTwister(prob.seed + 1000)
    n_mc = 100_000
    e_beta_L = mean(exp(prob.beta_cov[3] *
                        (100.0 * rand(rng_th, Beta(c.alpha_L, c.beta_L)) -
                         prob.z_bar[3])) for _ in 1:n_mc)
    e_f_theory = exp(prob.beta_cov[1]^2 * c.sigma_T^2 / 2) *
                 exp(c.lambda * (exp(prob.beta_cov[2]) - 1) -
                     prob.beta_cov[2] * c.lambda) * e_beta_L
    e_f_pool = mean(_cov_factor(prob.beta_cov, prob.z_bar, @view z[:, r])
                    for (z, _) in prob.pool for r in 1:N_R)
    dev_f = abs(e_f_pool - e_f_theory) / e_f_theory * 100
    @printf(io, "修正因子 E[f]: 理论 %.4f  池内 %.4f  偏差 %.2f%%\n",
            e_f_theory, e_f_pool, dev_f)
    println(io, "="^72)
    return nothing
end
