# run.jl — V2 连续基线的唯一入口：建模型，可选冒烟训练。
#
#   julia run.jl              # 用 config/problem.toml 建 T=60 的模型（不训练）
#   julia run.jl smoke        # T=2、池前 4 条，建模型 + SDDP.train 1 轮
#   julia run.jl smoke 5      # 同上，训练 5 轮
#
# 本文件不实现自定义训练/仿真/存取：训练用 SDDP.train，下界用
# SDDP.calculate_bound，仿真用 SDDP.simulate，割用 SDDP.write_cuts_to_file
# / SDDP.read_cuts_from_file。位置参数只在给了的时候才传入 main。
#
# 求解器：默认 COPT（与 Python V2 同族）；COPT.jl 不可用时自动回退 HiGHS。
# 这是求解器层选择，不改变模型方程。

const _ROOT = dirname(abspath(@__FILE__))
include(joinpath(_ROOT, "code", "problem_data.jl"))
include(joinpath(_ROOT, "code", "model.jl"))

function main(;
        config_path::AbstractString = default_problem_toml(),
        T::Union{Nothing,Int} = nothing,
        pool_size::Union{Nothing,Int} = nothing,
        seed::Union{Nothing,Int} = nothing,
        allow_julia_pool::Bool = false,
        iterations::Int = 0,
        optimizer = default_optimizer(),
    )
    prob = DefineProblem(;
        config_path = config_path,
        T = T,
        pool_size = pool_size,
        seed = seed,
        allow_julia_pool = allow_julia_pool,
    )
    print_summary(prob)

    model = build_sddp_model(prob; optimizer = optimizer)
    println("求解器：$(optimizer)")
    println("模型已建：$(prob.T + 1) 个节点（1 虚拟 + $(prob.T) 真实期），",
            "状态 $(state_dim(prob)) 维，场景池 $(length(prob.pool)) 条，纯 LP（0 整数变量）。")

    if iterations > 0
        SDDP.train(model; iteration_limit = iterations, print_level = 1)
        @printf("LB = %.6f\n", SDDP.calculate_bound(model))
    else
        println("未训练。内置 API：")
        println("  SDDP.train(model; iteration_limit = ..., print_level = 1)")
        println("  SDDP.calculate_bound(model)")
        println("  SDDP.simulate(model, n_paths; parallel_scheme = SDDP.Serial())")
        println("  SDDP.write_cuts_to_file(model, \"cuts.json\")")
    end
    return model, prob
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = String.(ARGS)
    if isempty(args)
        main()
    elseif args[1] == "smoke"
        it = length(args) >= 2 ? parse(Int, args[2]) : 1
        main(T = 2, pool_size = 4, iterations = it)
    else
        error("用法：\n  julia run.jl              # 建默认 T=60 模型（不训练）\n" *
              "  julia run.jl smoke [n]    # T=2、池 4 条、训练 n 轮（默认 1）")
    end
end
