# run_2h.jl — V2 Julia/SDDP.jl 2h 连续对偶训练 + 3000 路径仿真。
#
# 只调用 SDDP.jl 内置接口：
#   SDDP.train(..., duality_handler=ContinuousConicDuality)
#   SDDP.calculate_bound
#   SDDP.simulate
#   SDDP.write_cuts_to_file
#
# 用法：
#   julia -t 10 test/run_2h.jl [train_seconds] [sim_paths]
using Pkg
using Dates
using Printf
using Statistics
using JuMP
using SDDP
using COPT

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "code", "problem_data.jl"))
include(joinpath(ROOT, "code", "model.jl"))

function main(train_seconds::Float64 = 7200.0, sim_paths::Int = 3000)
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out = joinpath(ROOT, "results", "julia_v2_2h_$(stamp)")
    mkpath(out)
    cuts_path = joinpath(out, "model_cut.json")
    log_path = joinpath(out, "sddp_training_log.csv")
    summary_path = joinpath(out, "summary_$(stamp).txt")

    prob = DefineProblem()
    print_summary(prob)
    # 与 Python V2 的并行口径对齐：10 个 SDDP 线程，每个 COPT 子问题 1 线程。
    opt = optimizer_with_attributes(
        COPT.Optimizer, "Threads" => 1, "Logging" => 0)
    model = build_sddp_model(prob; optimizer = opt)
    println("求解器：COPT.Optimizer，Threads=1；SDDP 线程数=", Threads.nthreads())
    println("结果目录：", out)

    t0 = time()
    last_save = Ref(t0)
    function autosave(result)
        if time() - last_save[] >= 600.0
            SDDP.write_cuts_to_file(model, cuts_path)
            last_save[] = time()
            @printf("  autosave cuts at t=%.1fs, bound=%.6f\n",
                    time() - t0, result.bound)
        end
        return nothing
    end

    SDDP.train(
        model;
        iteration_limit = 100000,
        time_limit = train_seconds,
        duality_handler = SDDP.ContinuousConicDuality(),
        cut_type = SDDP.SINGLE_CUT,
        log_file = log_path,
        log_frequency = 1,
        print_level = 1,
        parallel_scheme = SDDP.Threaded(),
        post_iteration_callback = autosave,
    )
    train_wall = time() - t0
    lb = SDDP.calculate_bound(model)
    SDDP.write_cuts_to_file(model, cuts_path)
    println(@sprintf("训练结束：wall=%.1fs  LB=%.6f", train_wall, lb))

    sims = SDDP.simulate(model, sim_paths; parallel_scheme = SDDP.Threaded())
    costs = [sum(stage[:stage_objective] for stage in path) for path in sims]
    mu = mean(costs)
    sd = std(costs)
    ci = 1.959963984540054 * sd / sqrt(length(costs))
    gap = (mu - lb) / abs(mu) * 100.0
    println(@sprintf("%d 路径 UB=%.6f ± %.6f  gap=%.6f%%",
                     sim_paths, mu, ci, gap))

    open(summary_path, "w") do io
        println(io, "模型: V2 连续等级模型 (Julia SDDP.jl)")
        println(io, "训练时限 s: ", train_seconds)
        println(io, "训练墙钟 s: ", train_wall)
        println(io, "SDDP 线程: ", Threads.nthreads())
        println(io, "求解器: COPT.Optimizer Threads=1")
        println(io, "割类型: SINGLE_CUT")
        println(io, "对偶处理器: ContinuousConicDuality")
        println(io, "LB: ", lb)
        println(io, "仿真路径: ", sim_paths)
        println(io, "UB: ", mu)
        println(io, "std: ", sd)
        println(io, "95% CI: ", ci)
        println(io, "gap_percent: ", gap)
        println(io, "cuts: ", cuts_path)
        println(io, "log: ", log_path)
    end
    println("summary: ", summary_path)
    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    secs = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 7200.0
    nsim = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3000
    main(secs, nsim)
end
