# run.jl — minimal build entry for the Tier2 full-integer model.
#
#   julia run.jl                       # load the Tier2 config, build, print summary
#   julia run.jl smoke                 # T=2, pool=4, build only
#   julia run.jl train 1               # build + one SDDP.train iteration (Tier2 pool)
#   julia run.jl smoke_train 1         # smoke + one SDDP.train iteration
#
# Training is deliberately only the built-in SDDP.jl call. There is no custom
# training loop, simulation, cut save/load or cut enhancement in this project.
#
# The default optimizer is COPT.Optimizer (same as the Python/reference
# instance). HiGHS remains available for a cross-check:
#     include("run.jl"); main(optimizer = HiGHS.Optimizer)
#
# Duality handler:
#   :continuous (default) -> SDDP.ContinuousConicDuality()
#       This is the Python-Tier2 backward pass: forward MIP, backward LP
#       relaxation, hence valid Benders lower bounds.
#   :lagrangian -> SDDP.LagrangianDuality()
#       Optional SDDiP-style handler; useful if the LP relaxation cuts are too
#       weak, but not the Python-Tier2 protocol.

const _ROOT = dirname(abspath(@__FILE__))
include(joinpath(_ROOT, "code", "problem_data.jl"))
include(joinpath(_ROOT, "code", "model.jl"))

function main(;
    config::AbstractString = joinpath(_ROOT, "config", "problem.toml"),
    data::Union{Nothing,AbstractString} = default_problem_data_path(),
    T::Union{Nothing,Int} = nothing,
    pool_size::Union{Nothing,Int} = nothing,
    seed::Union{Nothing,Int} = nothing,
    smoke::Bool = false,
    iterations::Int = 0,
    duality::Symbol = :continuous,
    lagrangian_steps::Int = 5,
    print_level::Int = 1,
    threaded::Bool = false,
    optimizer = COPT.Optimizer,
)
    if smoke
        T = something(T, 2)
        pool_size = something(pool_size, 4)
        println("smoke overrides: T=$(T), pool_size=$(pool_size)" *
                "（导出池存在时取前 $(pool_size) 条，否则 Julia 采样）")
    end
    prob = DefineProblem(config;
        T = T, pool_size = pool_size, seed = seed, data_path = data)
    print_summary(prob)
    model = build_sddp_model(prob; optimizer = optimizer)
    println("model built: stages = $(prob.T + 1)（含虚拟首阶段），" *
            "pool = $(length(prob.pool))，状态维 = $(state_dim(prob))，" *
            "optimizer = $(optimizer)")

    if iterations > 0
        handler = if duality === :continuous
            SDDP.ContinuousConicDuality()
        elseif duality === :lagrangian
            SDDP.LagrangianDuality(
                method = SDDP.LocalImprovementSearch.BFGS(lagrangian_steps))
        else
            error("duality 只能为 :continuous / :lagrangian，收到 :$(duality)")
        end
        scheme = threaded && Threads.nthreads() > 1 ? SDDP.Threaded() : SDDP.Serial()
        println("SDDP.train: iterations=$(iterations), duality=:$(duality), " *
                "parallel=$(typeof(scheme))")
        if prob.T > 10 && length(prob.pool) > 20
            println("注意：全池/T=$(prob.T) 的单轮后向会枚举 " *
                    "$(length(prob.pool)) 场景 × $(prob.T) 阶段；快速验证请用 smoke_train。")
        end
        SDDP.train(model;
            iteration_limit = iterations,
            print_level = print_level,
            duality_handler = handler,
            parallel_scheme = scheme,
            run_numerical_stability_report = false,
        )
        lb = SDDP.calculate_bound(model)
        println("LB (SDDP.calculate_bound) = $(lb)")
    end
    return (model = model, prob = prob)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = String.(ARGS)
    if isempty(args)
        main()
    elseif args[1] == "smoke"
        main(smoke = true)
    elseif args[1] == "train"
        isempty(args[2:end]) && error("用法：julia run.jl train <iterations>")
        main(iterations = parse(Int, args[2]))
    elseif args[1] == "smoke_train"
        isempty(args[2:end]) && error("用法：julia run.jl smoke_train <iterations>")
        main(smoke = true, iterations = parse(Int, args[2]))
    else
        error("未知参数 $(args[1])；用法：julia run.jl [smoke|train <n>|smoke_train <n>]")
    end
end
