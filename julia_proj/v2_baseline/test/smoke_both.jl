# smoke_both.jl — 同一 T=2/pool=4 冒烟，COPT vs HiGHS 各训练 1 轮。
# 用同一随机种子重放前向路径，比较一轮后的 LB；对偶差异见 compare_stage.jl。
# 只调用 SDDP.jl 内置训练。
include(joinpath(@__DIR__, "..", "run.jl"))
using Random

prob = DefineProblem(T = 2, pool_size = 4)
println("COPT available: ", HAS_COPT)
for (name, opt) in (("COPT", default_optimizer()), ("HiGHS", HiGHS.Optimizer))
    model = build_sddp_model(prob; optimizer = opt)
    Random.seed!(2026)
    SDDP.train(model; iteration_limit = 1, print_level = 0)
    @printf("%-6s LB = %.10f\n", name, SDDP.calculate_bound(model))
end
