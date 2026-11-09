# problem_data.jl — Tier2 grade model data (Python-authoritative).
#
# Source of truth for equations/parameters:
#   code/problem_data.py, code/state_layout.py, code/transition.py,
#   code/subproblem.py, config/problem.toml, docs/建模与实现.md
#
# This project is the Tier2 full-integer variant:
#   - T=60, G=4, N_R=5, L=3;
#   - state vector I | o | n | q; n0 integer via largest remainder;
#   - N flows on valid arcs (j >= i), integer; Nsum + cumulative-tail bands;
#   - Theta enters only as a coefficient of n.in in the tail rows.
#
# The Python pool (z, Theta) is loaded from an exported problem_data.json so the
# stochastic input is bit-identical to the Python Tier2 instance. If the file is
# absent, a Julia-sampled pool is available as a fallback (not bit-identical).

using Random, Statistics, Printf, TOML, JSON
import LinearAlgebra
using Distributions: Normal, Poisson, Beta, Dirichlet

# ---------------------------------------------------------------- state layout
"""State vector layout I(N_R) | o((L−1)·N_R) | n(N_R·G) | q(N_R).

Indices are 1-based; `off_*` is the number of components before the block.
"""
struct StateLayout
    N_R::Int
    G::Int
    L::Int
    off_I::Int
    off_o::Int
    off_n::Int
    off_q::Int
    dim::Int
end

function StateLayout(N_R::Int, G::Int, L::Int)
    L >= 2 || error("Tier2 要求 L ≥ 2（在途管 L−1 ≥ 1），实际 $(L)")
    off_I = 0
    off_o = N_R
    off_n = off_o + (L - 1) * N_R
    off_q = off_n + N_R * G
    return StateLayout(N_R, G, L, off_I, off_o, off_n, off_q, off_q + N_R)
end

ix_I(Ly::StateLayout, r::Int) = Ly.off_I + r
ix_o(Ly::StateLayout, l::Int, r::Int) = Ly.off_o + (l - 1) * Ly.N_R + r
ix_n(Ly::StateLayout, r::Int, i::Int) = Ly.off_n + (r - 1) * Ly.G + i
ix_q(Ly::StateLayout, r::Int) = Ly.off_q + r

# ---------------------------------------------------------------- config
"""TOML problem configuration, mirroring Python's ProblemConfig (problem.toml)."""
struct ProblemConfig
    T::Int
    G::Int
    N_R::Int
    L::Int
    N::Vector{Int}
    Qmax::Vector{Int}
    Imax::Vector{Int}
    fleet_tol::Float64
    state_round_digits::Int
    state_repair_tol::Float64
    init_mix::Vector{Float64}
    n0_mode::String
    I0::Vector{Float64}
    C::Float64
    h::Float64
    Cs::Float64
    atc::Float64
    tc::Float64
    lambda0::Float64
    dirichlet_s::Float64
    p_digits::Int
    beta_cov::Vector{Float64}
    z_bar::Vector{Float64}
    cov::Dict{String,Float64}
    pool_size::Int
    seed::Int
    priority::Union{Nothing,Dict{Int,Vector{Int}}}
end

function _req(d::AbstractDict, key::AbstractString, where::AbstractString)
    haskey(d, key) || error(
        "配置 $(where) 缺少必填项 '$(key)'；默认值只写在 config/problem.toml，代码不兜底")
    return d[key]
end

function _section(raw::AbstractDict, name::AbstractString, where::AbstractString)
    sec = _req(raw, name, where)
    sec isa AbstractDict ||
        error("配置 $(where) 的 [$(name)] 必须是表，实际 $(typeof(sec))")
    return sec
end

"""Read `problem.toml` and validate the Tier2 invariants. Defaults are not
duplicated in code: every structural key must be present in the TOML."""
function load_problem_config(path::AbstractString)
    raw = TOML.parsefile(path)
    br = basename(path)
    dim = _section(raw, "dimension", br)
    fleet = _section(raw, "fleet", br)
    cost = _section(raw, "cost", br)
    deg = _section(raw, "degradation", br)
    cov = _section(raw, "covariate", br)
    scen = _section(raw, "scenario", br)

    T = Int(_req(dim, "T", "$(br)/dimension"))
    G = Int(_req(dim, "G", "$(br)/dimension"))
    N_R = Int(_req(dim, "N_R", "$(br)/dimension"))
    L = Int(_req(dim, "L", "$(br)/dimension"))
    N = Int.(_req(fleet, "N", "$(br)/fleet"))
    Qmax = Int.(_req(fleet, "Qmax", "$(br)/fleet"))
    Imax = Int.(_req(fleet, "Imax", "$(br)/fleet"))
    fleet_tol = Float64(_req(fleet, "fleet_tol", "$(br)/fleet"))
    state_round_digits = Int(_req(fleet, "state_round_digits", "$(br)/fleet"))
    state_repair_tol = Float64(_req(fleet, "state_repair_tol", "$(br)/fleet"))
    init_mix = Float64.(_req(fleet, "init_mix", "$(br)/fleet"))
    # Tier2 requires the integer initial mix; do not silently default to the
    # continuous V2口径.
    n0_mode = String(_req(fleet, "n0_mode", "$(br)/fleet"))
    I0 = Float64.(_req(fleet, "I0", "$(br)/fleet"))
    C = Float64(_req(cost, "C", "$(br)/cost"))
    h = Float64(_req(cost, "h", "$(br)/cost"))
    Cs = Float64(_req(cost, "Cs", "$(br)/cost"))
    atc = Float64(_req(cost, "atc", "$(br)/cost"))
    tc = Float64(_req(cost, "tc", "$(br)/cost"))
    lambda0 = Float64(_req(deg, "lambda0", "$(br)/degradation"))
    dirichlet_s = Float64(_req(deg, "dirichlet_s", "$(br)/degradation"))
    p_digits = Int(_req(deg, "p_digits", "$(br)/degradation"))
    beta_cov = Float64.(_req(deg, "beta_cov", "$(br)/degradation"))
    z_bar = Float64.(_req(deg, "z_bar", "$(br)/degradation"))
    covd = Dict{String,Float64}(k => Float64(_req(cov, k, "$(br)/covariate"))
                                for k in ("mu_T", "sigma_T", "lambda",
                                          "alpha_L", "beta_L"))
    pool_size = Int(_req(scen, "pool_size", "$(br)/scenario"))
    seed = Int(_req(scen, "seed", "$(br)/scenario"))
    prio_raw = get(raw, "priority", nothing)
    priority = nothing
    if prio_raw !== nothing
        priority = Dict{Int,Vector{Int}}()
        for (k, v) in prio_raw
            priority[parse(Int, k)] = Int.(v)
        end
    end

    # ---- validation (mirrors Python ProblemConfig._check) ----
    T > 0 || error("[dimension] T 必须为正，实际 $(T)")
    G >= 2 || error("[dimension] G 必须 ≥ 2，实际 $(G)")
    N_R > 0 || error("[dimension] N_R 必须为正，实际 $(N_R)")
    L >= 2 || error("[dimension] 要求 L ≥ 2，实际 $(L)")
    for (name, v) in (("N", N), ("Qmax", Qmax), ("Imax", Imax))
        length(v) == N_R ||
            error("[fleet] $(name) 长度 $(length(v)) ≠ N_R=$(N_R)")
        any(v .< 0) && error("[fleet] $(name) 不得含负值：$(v)")
    end
    any(Qmax .== 0) && error("[fleet] Qmax 不得含 0：$(Qmax)")
    any(N .== 0) && error("[fleet] N 不得含 0：$(N)")
    0.0 <= fleet_tol <= 1e-3 ||
        error("[fleet] fleet_tol 须在 [0, 1e-3]，实际 $(fleet_tol)")
    0 <= state_round_digits <= 12 ||
        error("[fleet] state_round_digits 须在 0..12，实际 $(state_round_digits)")
    fleet_tol <= state_repair_tol <= 1e-2 ||
        error("[fleet] state_repair_tol 须在 [fleet_tol, 1e-2]，实际 $(state_repair_tol)")
    length(init_mix) == G ||
        error("[fleet] init_mix 长度 $(length(init_mix)) ≠ G=$(G)")
    any(init_mix .< 0) && error("[fleet] init_mix 不得含负值：$(init_mix)")
    abs(sum(init_mix) - 1.0) <= 1e-8 ||
        error("[fleet] init_mix 必须和为 1，实际 $(sum(init_mix))")
    n0_mode == "integer_largest_remainder" ||
        error("[fleet] Tier2 要求 n0_mode=\"integer_largest_remainder\"，实际 \"$(n0_mode)\"")
    length(I0) == N_R || error("[fleet] I0 长度 $(length(I0)) ≠ N_R=$(N_R)")
    any(I0 .< 0) && error("[fleet] I0 不得含负值：$(I0)")
    any(I0 .> Imax) && error("[fleet] I0 超过 Imax：I0=$(I0), Imax=$(Imax)")
    any(abs.(I0 .- round.(I0)) .> 1e-9) &&
        error("[fleet] Tier2 要求整数 I0，实际 $(I0)")
    lambda0 > 0 || error("[degradation] lambda0 必须为正，实际 $(lambda0)")
    dirichlet_s > 0 || error("[degradation] dirichlet_s 必须为正，实际 $(dirichlet_s)")
    0 <= p_digits <= 6 || error("[degradation] p_digits 须在 0..6，实际 $(p_digits)")
    length(beta_cov) == 3 && length(z_bar) == 3 ||
        error("[degradation] PHM 要求 3 维协变量，实际 beta_cov=$(beta_cov), z_bar=$(z_bar)")
    for k in ("sigma_T", "lambda", "alpha_L", "beta_L")
        covd[k] > 0 || error("[covariate] $(k) 必须为正，实际 $(covd[k])")
    end
    pool_size >= 2 || error("[scenario] pool_size 至少为 2，实际 $(pool_size)")
    if priority !== nothing
        sort(collect(keys(priority))) == collect(1:N_R) ||
            error("[priority] 必须恰覆盖仓库 1..N_R，实际键 $(sort(collect(keys(priority))))")
        all(sort(v) == collect(1:N_R) for v in values(priority)) ||
            error("[priority] 每个列表必须是 1..N_R 的排列")
    end

    return ProblemConfig(T, G, N_R, L, N, Qmax, Imax, fleet_tol,
                         state_round_digits, state_repair_tol, init_mix, n0_mode,
                         I0, C, h, Cs, atc, tc, lambda0, dirichlet_s, p_digits,
                         beta_cov, z_bar, covd, pool_size, seed, priority)
end

# ---------------------------------------------------------------- transitions
"""Poisson pmf p_k = e^{-λ} λ^k / k!."""
function poisson_pmf(lam::Float64, k::Int)
    lam >= 0 || error("Poisson 均值必须非负，实际 $(lam)")
    lam == 0.0 && return k == 0 ? 1.0 : 0.0
    return exp(-lam) * lam^k / factorial(k)
end

"""G×(G+1) upper-triangular Poisson-increment transition matrix.

Mirrors Python `transition_matrix`: non-diagonal entries are rounded to
`p_digits`; the diagonal absorbs the rounding error. The failure column is
P(D ≥ G−i). Row sums are normalised to 1 after cleaning.
"""
function transition_matrix(lam::Float64, G::Int, p_digits::Union{Nothing,Int})
    lam >= 0 || error("lambda 必须非负，实际 $(lam)")
    p = [poisson_pmf(lam, k) for k in 0:G]
    P = zeros(Float64, G, G + 1)
    for i in 1:G
        off = Float64[]
        # 0-based source i0 = i-1; 0-based targets i0+1..G.
        for j0 in i:(G - 1)          # non-failure targets i0+1 .. G-1
            k = j0 - (i - 1)         # 0-based increment
            push!(off, p[k + 1])
        end
        # failure target G: P(D >= G - (i-1)) = 1 - sum_{k=0}^{G-i} p_k
        push!(off, 1.0 - sum(p[1:(G - i + 1)]))
        if p_digits !== nothing
            off = round.(off; digits = p_digits)
            s = sum(off)
            if s > 1.0 + 1e-12
                off ./= s
            end
        end
        P[i, i + 1:G + 1] .= off
        P[i, i] = 1.0 - sum(off)
    end
    P = max.(P, 0.0)
    for i in 1:G
        rs = sum(P[i, :])
        rs > 0 || error("转移矩阵第 $(i) 行为空")
        P[i, :] ./= rs
    end
    return P
end

"""C[r,i,j] = Σ_{k=j}^{G+1} Θ[r,i,k] (1-based; failure column is G+1)."""
function tail_cdf(theta::AbstractArray{<:Real,3})
    N_R, G, GC = size(theta)
    GC == G + 1 || error("Θ 形状 $(size(theta)) 的最后一维应为 G+1=$(G + 1)")
    c = zeros(Float64, N_R, G, G + 1)
    for r in 1:N_R, i in 1:G
        c[r, i, G + 1] = theta[r, i, G + 1]
        # `j = G` is needed for the tail at j = G even when i < G.
        c[r, i, G] = theta[r, i, G] + c[r, i, G + 1]
        for j in (G - 1):-1:i
            c[r, i, j] = theta[r, i, j] + c[r, i, j + 1]
        end
    end
    return c
end

"""Largest-remainder integer allocation, same tie rule as Python
`largest_remainder_counts` (descending frac, then ascending index)."""
function largest_remainder_counts(total::Int, mix::AbstractVector{<:Real})
    total >= 0 || error("total 必须非负，实际 $(total)")
    any(mix .< 0) && error("mix 不得含负值：$(mix)")
    abs(sum(mix) - 1.0) <= 1e-8 || error("mix 必须和为 1，实际 $(sum(mix))")
    raw = total .* Float64.(mix)
    base = floor.(Int, raw)
    deficit = total - sum(base)
    deficit >= 0 || error("分配基数超过 total")
    frac = raw .- base
    eligible = findall(>(0.0), mix)
    deficit <= length(eligible) ||
        error("最大余数分配无法补齐：deficit=$(deficit), eligible=$(length(eligible))")
    order = sort(eligible, by = j -> (-frac[j], j))
    for j in order[1:deficit]
        base[j] += 1
    end
    sum(base) == total || error("最大余数分配后总和 $(sum(base)) ≠ $(total)")
    return base
end

# ---------------------------------------------------------------- priority
function default_priority(N_R::Int)
    if N_R == 5
        return Dict(1 => [1, 2, 3, 4, 5], 2 => [2, 1, 3, 4, 5],
                    3 => [3, 1, 2, 4, 5], 4 => [4, 1, 2, 3, 5],
                    5 => [5, 1, 2, 3, 4])
    end
    return Dict(r => vcat(r, setdiff(1:N_R, r)) for r in 1:N_R)
end

"""tc_route[r,m] = (position of m in r's priority list − 1)·tc."""
function make_tc_route(priority::Dict{Int,Vector{Int}}, tc::Float64, N_R::Int)
    route = zeros(Float64, N_R, N_R)
    for r in 1:N_R, (pos, m) in enumerate(priority[r])
        route[r, m] = (pos - 1) * tc
    end
    return route
end

resolve_priority(cfg::ProblemConfig) =
    cfg.priority === nothing ? default_priority(cfg.N_R) : cfg.priority

# ---------------------------------------------------------------- scenario pool
_cov_factor(beta_cov, z_bar, z_r) =
    exp(LinearAlgebra.dot(beta_cov, z_r .- z_bar))

"""Julia-sampled pool fallback; distributionally equal to Python but not
bit-identical (NumPy PCG64 vs Julia MersenneTwister)."""
function sample_pool(cfg::ProblemConfig, pool_size::Int, seed::Int)
    rng = MersenneTwister(seed)
    G, N_R = cfg.G, cfg.N_R
    pool = Vector{Tuple{Matrix{Float64},Array{Float64,3}}}(undef, pool_size)
    for s in 1:pool_size
        z = zeros(Float64, 3, N_R)
        for r in 1:N_R
            z[1, r] = rand(rng, Normal(cfg.cov["mu_T"], cfg.cov["sigma_T"]))
        end
        for r in 1:N_R
            z[2, r] = rand(rng, Poisson(cfg.cov["lambda"]))
        end
        for r in 1:N_R
            z[3, r] = 100.0 * rand(rng, Beta(cfg.cov["alpha_L"], cfg.cov["beta_L"]))
        end
        theta = zeros(Float64, N_R, G, G + 1)
        for r in 1:N_R
            f = _cov_factor(cfg.beta_cov, cfg.z_bar, z[:, r])
            P = transition_matrix(cfg.lambda0 * f, G, cfg.p_digits)
            for i in 1:G
                cols = [j for j in i:(G + 1) if P[i, j] > 0.0]
                isempty(cols) && error("P 第 $(i) 行无正支撑")
                alpha = cfg.dirichlet_s .* P[i, cols]
                vals = length(cols) == 1 ? [1.0] : rand(rng, Dirichlet(alpha))
                theta[r, i, cols] .= vals
            end
        end
        _validate_theta(theta; where = "sampled scene $(s)")
        pool[s] = (z, theta)
    end
    return pool
end

function _validate_theta(theta::AbstractArray{<:Real,3}; where::AbstractString = "")
    N_R, G, GC = size(theta)
    GC == G + 1 || error("$(where): Θ 最后一维 $(GC) ≠ G+1")
    for r in 1:N_R, i in 1:G
        s = sum(theta[r, i, :])
        abs(s - 1.0) <= 1e-10 ||
            error("$(where): Θ[$(r),$(i),:] 行和 $(s) ≠ 1")
        if i > 1
            maximum(abs.(theta[r, i, 1:(i - 1)])) <= 1e-12 ||
                error("$(where): Θ[$(r),$(i),:] 下三角非零")
        end
    end
    any(theta .< 0) && error("$(where): Θ 出现负值")
    return nothing
end

function _pool_mean_theta(pool)
    N_R, G, GC = size(pool[1][2])
    acc = zeros(Float64, N_R, G, GC)
    for (_, th) in pool
        acc .+= th
    end
    return acc ./ length(pool)
end

function _pool_mean_P(cfg::ProblemConfig, pool)
    G = cfg.G
    acc = zeros(Float64, G, G + 1)
    for (z, _) in pool, r in 1:cfg.N_R
        f = _cov_factor(cfg.beta_cov, cfg.z_bar, z[:, r])
        acc .+= transition_matrix(cfg.lambda0 * f, G, cfg.p_digits)
    end
    return acc ./ (length(pool) * cfg.N_R)
end

# ---------------------------------------------------------------- exported data
function default_problem_data_path()
    root = dirname(dirname(abspath(@__FILE__)))
    return joinpath(root, "data", "problem_data.json")
end

_tofloat(x) = Float64(x)
_toint(x) = Int(x)

function _json_matrix(d, key; rows::Int, cols::Int)
    raw = _req(d, key, "problem_data.json")
    M = zeros(Float64, rows, cols)
    for i in 1:rows, j in 1:cols
        M[i, j] = Float64(raw[i][j])
    end
    return M
end

function _json_vector(d, key; len::Int)
    raw = _req(d, key, "problem_data.json")
    length(raw) == len || error("problem_data.json[$(key)] 长度 $(length(raw)) ≠ $(len)")
    return Float64.(raw)
end

_check_num(name, got, want; atol = 1e-12, rtol = 1e-12) =
    isapprox(Float64(got), Float64(want); atol = atol, rtol = rtol) ||
        error("导出数据与配置不一致：$(name)，JSON=$(got)，配置=$(want)")

_check_int(name, got, want) =
    Int(got) == Int(want) || error("导出数据与配置不一致：$(name)，JSON=$(got)，配置=$(want)")

"""Load the Python-exported `problem_data.json` and cross-check every
parameter that the model uses. This prevents silent drift from the Python
instance (Tier2 semantics are Python-authoritative)."""
function load_exported_data(path::AbstractString, cfg::ProblemConfig)
    isfile(path) || error("找不到导出的 problem_data.json：$(path)")
    d = JSON.parsefile(path)
    get(d, "model", "") == "markov_grade" ||
        error("不是 markov_grade 的 problem_data.json：$(path)")

    _check_int("T", _req(d, "T", "problem_data.json"), cfg.T)
    _check_int("G", _req(d, "G", "problem_data.json"), cfg.G)
    _check_int("N_R", _req(d, "N_R", "problem_data.json"), cfg.N_R)
    _check_int("L", _req(d, "L", "problem_data.json"), cfg.L)
    _check_int("seed", _req(d, "seed", "problem_data.json"), cfg.seed)
    _check_int("p_digits", _req(d, "p_digits", "problem_data.json"), cfg.p_digits)
    _check_int("state_round_digits", _req(d, "state_round_digits", "problem_data.json"),
               cfg.state_round_digits)
    _check_num("fleet_tol", _req(d, "fleet_tol", "problem_data.json"), cfg.fleet_tol)
    _check_num("state_repair_tol", _req(d, "state_repair_tol", "problem_data.json"),
               cfg.state_repair_tol)
    _check_num("lambda0", _req(d, "lambda0", "problem_data.json"), cfg.lambda0)
    _check_num("dirichlet_s", _req(d, "dirichlet_s", "problem_data.json"),
               cfg.dirichlet_s)

    for (key, want) in (("N", cfg.N), ("Qmax", cfg.Qmax), ("Imax", cfg.Imax))
        got = Int.(_req(d, key, "problem_data.json"))
        got == want || error("导出数据与配置不一致：$(key)，JSON=$(got)，配置=$(want)")
    end
    for (key, want) in (("C", cfg.C), ("h", cfg.h), ("Cs", cfg.Cs),
                        ("atc", cfg.atc), ("tc", cfg.tc))
        _check_num(key, _req(d, key, "problem_data.json"), want)
    end
    _check_num("init_mix", sum(abs.(_json_vector(d, "init_mix"; len = cfg.G)
                                   .- cfg.init_mix)), 0.0; atol = 1e-12)
    _check_num("I0", sum(abs.(_json_vector(d, "I0"; len = cfg.N_R) .- cfg.I0)), 0.0;
               atol = 1e-12)
    _check_num("beta_cov",
               sum(abs.(_json_vector(d, "beta_cov"; len = 3) .- cfg.beta_cov)), 0.0;
               atol = 1e-12)
    _check_num("z_bar", sum(abs.(_json_vector(d, "z_bar"; len = 3) .- cfg.z_bar)), 0.0;
               atol = 1e-12)
    for k in ("mu_T", "sigma_T", "lambda", "alpha_L", "beta_L")
        _check_num("cov.$(k)", _req(_req(d, "cov", "problem_data.json"), k,
                                  "problem_data.json/cov"), cfg.cov[k])
    end
    get(d, "n0_mode", "") == cfg.n0_mode ||
        error("导出数据 n0_mode=$(get(d, "n0_mode", "")) ≠ 配置 $(cfg.n0_mode)")

    # P_ref validates our transition_matrix implementation.
    P_ref_json = _json_matrix(d, "P_ref"; rows = cfg.G, cols = cfg.G + 1)
    P_ref_cfg = transition_matrix(cfg.lambda0, cfg.G, cfg.p_digits)
    maximum(abs.(P_ref_json .- P_ref_cfg)) <= 1e-12 ||
        error("P_ref 与 Julia transition_matrix 不一致：max|Δ|=" *
              "$(maximum(abs.(P_ref_json .- P_ref_cfg)))")

    # n0 validates the largest-remainder implementation.
    n0_json = _json_matrix(d, "n0"; rows = cfg.N_R, cols = cfg.G)
    n0_cfg = Matrix{Int}(undef, cfg.N_R, cfg.G)
    for r in 1:cfg.N_R
        n0_cfg[r, :] = largest_remainder_counts(cfg.N[r], cfg.init_mix)
    end
    maximum(abs.(n0_json .- n0_cfg)) <= 1e-9 ||
        error("n0 与 Julia 最大余数分配不一致：JSON=$(n0_json)，配置=$(n0_cfg)")
    any(abs.(n0_json .- round.(n0_json)) .> 1e-9) &&
        error("导出 n0 不是整数")

    # Priority / route cost.
    prio_json = get(d, "priority", nothing)
    prio_cfg = resolve_priority(cfg)
    if prio_json !== nothing
        prio_j = Dict{Int,Vector{Int}}()
        for (k, v) in prio_json
            prio_j[parse(Int, k)] = Int.(v)
        end
        prio_j == prio_cfg ||
            error("priority 与 Julia 默认/配置优先表不一致：JSON=$(prio_j)，配置=$(prio_cfg)")
    end
    tc_route_json = _json_matrix(d, "tc_route"; rows = cfg.N_R, cols = cfg.N_R)
    tc_route_cfg = make_tc_route(prio_cfg, cfg.tc, cfg.N_R)
    maximum(abs.(tc_route_json .- tc_route_cfg)) <= 1e-12 ||
        error("tc_route 与 Julia 计算不一致")

    # Pool: exact copy of the Python `(z, Theta)` scenes, equal weights.
    pool_raw = _req(d, "pool", "problem_data.json")
    length(pool_raw) == cfg.pool_size ||
        error("problem_data.json pool 长度 $(length(pool_raw)) ≠ 配置 pool_size=$(cfg.pool_size)")
    pool = Vector{Tuple{Matrix{Float64},Array{Float64,3}}}(undef, length(pool_raw))
    for s in eachindex(pool_raw)
        item = pool_raw[s]
        length(item) == 2 || error("pool[$(s)] 不是 [z, Theta]")
        zraw, thraw = item[1], item[2]
        z = zeros(Float64, 3, cfg.N_R)
        for k in 1:3, r in 1:cfg.N_R
            z[k, r] = Float64(zraw[k][r])
        end
        th = zeros(Float64, cfg.N_R, cfg.G, cfg.G + 1)
        for r in 1:cfg.N_R, i in 1:cfg.G, j in 1:(cfg.G + 1)
            th[r, i, j] = Float64(thraw[r][i][j])
        end
        _validate_theta(th; where = "exported scene $(s)")
        pool[s] = (z, th)
    end
    return pool
end

# ---------------------------------------------------------------- ProblemData
struct ProblemData
    T::Int
    G::Int
    N_R::Int
    L::Int
    N::Vector{Int}
    n0::Matrix{Int}
    n0_mode::String
    I0::Vector{Float64}
    o0::Matrix{Float64}
    init_mix::Vector{Float64}
    C::Float64
    h::Float64
    Cs::Float64
    atc::Float64
    tc::Float64
    Qmax::Vector{Int}
    Imax::Vector{Int}
    fleet_tol::Float64
    state_round_digits::Int
    state_repair_tol::Float64
    priority::Dict{Int,Vector{Int}}
    tc_route::Matrix{Float64}
    lambda0::Float64
    dirichlet_s::Float64
    p_digits::Int
    beta_cov::Vector{Float64}
    z_bar::Vector{Float64}
    cov::Dict{String,Float64}
    P_ref::Matrix{Float64}
    pool::Vector{Tuple{Matrix{Float64},Array{Float64,3}}}
    seed::Int
    pool_source::String
end

state_dim(prob::ProblemData) = StateLayout(prob.N_R, prob.G, prob.L).dim
state_layout(prob::ProblemData) = StateLayout(prob.N_R, prob.G, prob.L)

function _pool_mean_P(prob::ProblemData, pool)
    acc = zeros(Float64, prob.G, prob.G + 1)
    for (z, _) in pool, r in 1:prob.N_R
        f = _cov_factor(prob.beta_cov, prob.z_bar, z[:, r])
        acc .+= transition_matrix(prob.lambda0 * f, prob.G, prob.p_digits)
    end
    return acc ./ (length(pool) * prob.N_R)
end

"""Upper bounds in the Python state order I|o|n|q."""
function state_upper(prob::ProblemData)
    Ly = state_layout(prob)
    ub = zeros(Float64, Ly.dim)
    for r in 1:prob.N_R
        ub[ix_I(Ly, r)] = prob.Imax[r]
        for l in 1:(prob.L - 1)
            ub[ix_o(Ly, l, r)] = prob.Qmax[r]
        end
        for i in 1:prob.G
            ub[ix_n(Ly, r, i)] = prob.N[r]
        end
        ub[ix_q(Ly, r)] = prob.Qmax[r]
    end
    return ub
end

"""Initial state in the Python order I|o|n|q."""
function initial_state(prob::ProblemData)
    Ly = state_layout(prob)
    x = zeros(Float64, Ly.dim)
    for r in 1:prob.N_R
        x[ix_I(Ly, r)] = prob.I0[r]
        for l in 1:(prob.L - 1)
            x[ix_o(Ly, l, r)] = prob.o0[l, r]
        end
        for i in 1:prob.G
            x[ix_n(Ly, r, i)] = prob.n0[r, i]
        end
    end
    return x
end

function DefineProblem(config_path::AbstractString;
                       T::Union{Nothing,Int} = nothing,
                       pool_size::Union{Nothing,Int} = nothing,
                       seed::Union{Nothing,Int} = nothing,
                       data_path::Union{Nothing,AbstractString} =
                           default_problem_data_path(),
                       pool_source::Symbol = :auto)
    cfg = load_problem_config(config_path)
    # Smoke overrides only change the horizon and the number of pool scenes;
    # all equations/parameters remain the Tier2 instance.
    T_eff = T === nothing ? cfg.T : Int(T)
    T_eff > 0 || error("T 必须为正，实际 $(T_eff)")
    pool_size_eff = pool_size === nothing ? cfg.pool_size : Int(pool_size)
    pool_size_eff >= 2 || error("pool_size 至少为 2，实际 $(pool_size_eff)")
    seed_eff = seed === nothing ? cfg.seed : Int(seed)

    priority = resolve_priority(cfg)
    tc_route = make_tc_route(priority, cfg.tc, cfg.N_R)
    n0 = Matrix{Int}(undef, cfg.N_R, cfg.G)
    for r in 1:cfg.N_R
        n0[r, :] = largest_remainder_counts(cfg.N[r], cfg.init_mix)
    end
    o0 = zeros(Float64, cfg.L - 1, cfg.N_R)
    P_ref = transition_matrix(cfg.lambda0, cfg.G, cfg.p_digits)

    pool = nothing
    src = ""
    use_exported = pool_source === :exported ||
                   (pool_source === :auto && data_path !== nothing && isfile(data_path))
    if use_exported
        data_path === nothing && error("pool_source=:exported 需要 data_path")
        if seed !== nothing && Int(seed) != cfg.seed
            @warn "导出池模式忽略 seed 覆盖：池来自 problem_data.json" maxlog = 1
        end
        seed_eff = cfg.seed   # ProblemData.seed 记录生成导出池的种子
        pool = load_exported_data(data_path, cfg)
        if pool_size_eff != length(pool)
            pool_size_eff > length(pool) &&
                error("pool_size=$(pool_size_eff) 超过导出池 $(length(pool))")
            pool = pool[1:pool_size_eff]
        end
        src = "exported:$(abspath(data_path))"
    elseif pool_source === :sampled ||
           (pool_source === :auto && (data_path === nothing || !isfile(data_path)))
        @warn "未加载导出的 problem_data.json，改用 Julia 采样池；与 Python 池不是逐位一致" maxlog = 1
        pool = sample_pool(cfg, pool_size_eff, seed_eff)
        src = "sampled:seed=$(seed_eff)"
    else
        error("未知 pool_source=$(pool_source)")
    end

    # n0 / I0 self-checks.
    for r in 1:cfg.N_R
        sum(n0[r, :]) == cfg.N[r] ||
            error("仓库 $(r) 的 n0 和 $(sum(n0[r, :])) ≠ N=$(cfg.N[r])")
        cfg.I0[r] <= cfg.Imax[r] ||
            error("仓库 $(r) 的 I0=$(cfg.I0[r]) 超过 Imax=$(cfg.Imax[r])")
    end

    return ProblemData(T_eff, cfg.G, cfg.N_R, cfg.L, copy(cfg.N), n0,
                       cfg.n0_mode, copy(cfg.I0), o0, copy(cfg.init_mix),
                       cfg.C, cfg.h, cfg.Cs, cfg.atc, cfg.tc,
                       copy(cfg.Qmax), copy(cfg.Imax), cfg.fleet_tol,
                       cfg.state_round_digits, cfg.state_repair_tol,
                       priority, tc_route, cfg.lambda0, cfg.dirichlet_s,
                       cfg.p_digits, copy(cfg.beta_cov), copy(cfg.z_bar),
                       copy(cfg.cov), P_ref, pool, seed_eff, src)
end

# ---------------------------------------------------------------- summary
function _cumulative_failure_prob(P::AbstractMatrix, T::Int, mix)
    G = size(P, 1)
    Q = P[:, 1:G]
    f = P[:, G + 1]
    v = Float64.(copy(mix))
    acc = 0.0
    for _ in 1:T
        acc += LinearAlgebra.dot(v, f)
        v = Q' * v
    end
    return acc
end

function _expected_first_passage(P::AbstractMatrix, mix)
    G = size(P, 1)
    Q = P[:, 1:G]
    M = (Matrix{Float64}(LinearAlgebra.I, G, G) - Q) \ ones(G)
    return LinearAlgebra.dot(Float64.(mix), M)
end

"""Print parameter summary and self-checks (pool row sums, structural zeros,
E[Θ] vs E_z[P], n0 integer, cumulative failure)."""
function print_summary(prob::ProblemData; io::IO = stdout)
    G, N_R = prob.G, prob.N_R
    println(io, "="^72)
    println(io, "问题数据摘要（Tier2 全整数等级模型，Python 为唯一事实源）")
    println(io, "="^72)
    println(io, "T = $(prob.T)  G = $(G)  N_R = $(N_R)  L = $(prob.L)  " *
                "pool = $(length(prob.pool))  seed = $(prob.seed)  " *
                "状态维 = $(state_dim(prob))")
    println(io, "pool_source = $(prob.pool_source)")
    println(io, "机队 N = $(prob.N)  总数 = $(sum(prob.N))  " *
                "init_mix = $(prob.init_mix)  n0_mode = $(prob.n0_mode)")
    println(io, "n0（整数最大余数） = $(prob.n0)")
    println(io, "I0 = $(prob.I0)  fleet_tol = $(prob.fleet_tol)  " *
                "state_round_digits = $(prob.state_round_digits)  " *
                "state_repair_tol = $(prob.state_repair_tol)")
    println(io, "成本: C=$(prob.C)  h=$(prob.h)  Cs=$(prob.Cs)  " *
                "atc=$(prob.atc)  tc=$(prob.tc)")
    println(io, "λ0=$(prob.lambda0)  dirichlet_s=$(prob.dirichlet_s)  " *
                "p_digits=$(prob.p_digits)")
    println(io, "PHM: beta_cov=$(prob.beta_cov)  z_bar=$(prob.z_bar)")
    c = prob.cov
    println(io, "协变量: 油温 N($(c["mu_T"]), $(c["sigma_T"])²)  " *
                "启停 Poisson($(c["lambda"]))  " *
                "满负荷 100·Beta($(c["alpha_L"]), $(c["beta_L"])) (%)")
    println(io, "P_ref（λ0 下，$(prob.p_digits) 位小数）:")
    for i in 1:G
        @printf(io, "  %s   sum=%.6f\n",
                join([@sprintf("%.6g", prob.P_ref[i, j]) for j in 1:(G + 1)], " "),
                sum(prob.P_ref[i, :]))
    end
    P_exact = transition_matrix(prob.lambda0, G, nothing)
    @printf(io, "P_ref vs 精确矩阵: max|Δ| = %.3e\n",
            maximum(abs.(prob.P_ref .- P_exact)))
    pT = _cumulative_failure_prob(prob.P_ref, prob.T, prob.init_mix)
    mttf = _expected_first_passage(prob.P_ref, prob.init_mix)
    @printf(io, "T=%d 期累计失效 = %.4f%%；首达约 %.1f 月\n", prob.T, 100 * pT, mttf)

    # Pool self-checks.
    max_row = 0.0
    max_zero = 0.0
    for (_, th) in prob.pool, r in 1:N_R, i in 1:G
        max_row = max(max_row, abs(sum(th[r, i, :]) - 1.0))
        if i > 1
            max_zero = max(max_zero, maximum(abs.(th[r, i, 1:(i - 1)])))
        end
    end
    theta_bar = _pool_mean_theta(prob.pool)
    Pz_bar = _pool_mean_P(prob, prob.pool)
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

    # Covariate-factor centring (pool-level diagnostic; no hard error for tiny
    # smoke pools).
    rng = MersenneTwister(prob.seed + 1000)
    n_mc = 20_000
    e_beta_L = mean(exp(prob.beta_cov[3] *
                        (100.0 * rand(rng, Beta(c["alpha_L"], c["beta_L"])) -
                         prob.z_bar[3])) for _ in 1:n_mc)
    e_f_theory = exp(prob.beta_cov[1]^2 * c["sigma_T"]^2 / 2) *
                 exp(c["lambda"] * (exp(prob.beta_cov[2]) - 1) -
                     prob.beta_cov[2] * c["lambda"]) * e_beta_L
    e_f_pool = mean(_cov_factor(prob.beta_cov, prob.z_bar, z[:, r])
                    for (z, _) in prob.pool for r in 1:N_R)
    @printf(io, "修正因子 E[f]: 理论 %.4f  池内 %.4f  偏差 %.2f%%\n",
            e_f_theory, e_f_pool, 100 * abs(e_f_pool - e_f_theory) / e_f_theory)
    println(io, "="^72)
    return nothing
end
