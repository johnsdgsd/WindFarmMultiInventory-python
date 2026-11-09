# validate_pool.jl — the full 300-scene pool / P_ref / transition function as
# loaded in memory by Julia must match the Python fingerprint.
#
# Run:
#   "D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/dump_pool_fingerprint.py
#   julia test/validate_pool.jl

const _ROOT = dirname(dirname(abspath(@__FILE__)))
include(joinpath(_ROOT, "code", "problem_data.jl"))
using JSON

const PYF = joinpath(_ROOT, "test", "_python_pool_fingerprint.json")
isfile(PYF) || error("先运行 test/dump_pool_fingerprint.py")
py = JSON.parsefile(PYF)

let
prob = DefineProblem(joinpath(_ROOT, "config", "problem.toml"))  # T=60, pool=300

println("="^80)
println("池 / P_ref / 过渡矩阵指纹（Python fixture vs Julia 内存）")
println("="^80)
println("pool_source = $(prob.pool_source)")
println("T=$(prob.T) pool=$(length(prob.pool)) seed=$(prob.seed)")
@assert startswith(prob.pool_source, "exported")
@assert prob.T == 60 && length(prob.pool) == 300
@assert prob.seed == 2026

# Scene 0/1/7 Theta and z.
th0 = prob.pool[1][2]
th1 = prob.pool[2][2]
th7 = prob.pool[8][2]
d0 = maximum(abs.(th0 .- [Float64(py["scene0_theta"][r][i][j])
                          for r in 1:prob.N_R, i in 1:prob.G,
                          j in 1:(prob.G + 1)]))
d1 = maximum(abs.(th1 .- [Float64(py["scene1_theta"][r][i][j])
                          for r in 1:prob.N_R, i in 1:prob.G,
                          j in 1:(prob.G + 1)]))
d7 = maximum(abs.(th7 .- [Float64(py["scene7_theta"][r][i][j])
                          for r in 1:prob.N_R, i in 1:prob.G,
                          j in 1:(prob.G + 1)]))
dz = maximum(abs.(prob.pool[1][1] .- [Float64(py["scene0_z"][k][r]) for k in 1:3,
                                      r in 1:prob.N_R]))
println("scene 0/1/7 Theta max|d| = $(maximum((d0, d1, d7)))，scene 0 z max|d|=$(dz)")
@assert maximum((d0, d1, d7)) == 0.0 && dz == 0.0

# Pool means.
mean_th = _pool_mean_theta(prob.pool)
mean_P = _pool_mean_P(prob, prob.pool)
dmt = maximum(abs.(mean_th .- [Float64(py["mean_theta"][r][i][j])
                               for r in 1:prob.N_R, i in 1:prob.G,
                               j in 1:(prob.G + 1)]))
dmp = maximum(abs.(mean_P .- [Float64(py["mean_P"][i][j]) for i in 1:prob.G,
                              j in 1:(prob.G + 1)]))
println("mean Theta max|d|=$(dmt)，mean E_z[P] max|d|=$(dmp)")
@assert dmt == 0.0 && dmp <= 1e-14

# Row sums / structural zeros / P_ref.
row_max = 0.0
zero_max = 0.0
n_tiny = 0
max_tiny = 0.0
for (_, th) in prob.pool
    for r in 1:prob.N_R, i in 1:prob.G
        row_max = max(row_max, abs(sum(th[r, i, :]) - 1.0))
        if i > 1
            zero_max = max(zero_max, maximum(abs.(th[r, i, 1:(i - 1)])))
        end
    end
    c = tail_cdf(th)
    for r in 1:prob.N_R, i in 1:prob.G, j in (i + 1):(prob.G + 1)
        v = abs(c[r, i, j])
        if 0.0 < v < 1e-10
            n_tiny += 1
            max_tiny = max(max_tiny, v)
        end
    end
end
println("Θ 行和 max|d|=$(row_max)，结构零 max|d|=$(zero_max)")
println("Tail 系数在 (0,1e-10)：$(n_tiny) 个，max=$(max_tiny)（Python: " *
        "$(py["n_tail_coeff_below_1e-10"]) / $(py["max_tail_coeff_below_1e-10"])）")
@assert row_max <= 1e-12 && zero_max == 0.0
@assert n_tiny == Int(py["n_tail_coeff_below_1e-10"])
@assert abs(max_tiny - Float64(py["max_tail_coeff_below_1e-10"])) <= 1e-18
dP = maximum(abs.(prob.P_ref .- [Float64(py["P_ref"][i][j]) for i in 1:prob.G,
                                 j in 1:(prob.G + 1)]))
println("P_ref max|d|=$(dP)")
@assert dP == 0.0

# Transition function at several lambdas (only used by P_ref / fallback).
for lam in py["transition_lambdas"]
    Pj = transition_matrix(Float64(lam), prob.G, prob.p_digits)
    Pp = [Float64(py["transition"][string(lam)][i][j]) for i in 1:prob.G,
          j in 1:(prob.G + 1)]
    d = maximum(abs.(Pj .- Pp))
    println("  transition_matrix(λ=$(lam)) max|d|=$(d)")
    @assert d == 0.0
end

println("POOL/TRANSITION: PASS")
end
