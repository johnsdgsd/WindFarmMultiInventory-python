# run_local_bandit.jl — 本机 Tier2 Julia 训练：COPT + BanditDuality + 2h + 3000 仿真。
#
# 只调用 SDDP.jl 内置接口；模型方程来自 code/model.jl，数据来自 Python 导出的 Tier2 配置/池。
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
    out = joinpath(ROOT, "results", "local_tier2_bandit_$(stamp)")
    mkpath(out)
    cuts_path = joinpath(out, "model_cut.json")
    log_path = joinpath(out, "sddp_training_log.csv")
    summary_path = joinpath(out, "summary_$(stamp).txt")
    prob = DefineProblem(joinpath(ROOT, "config", "problem.toml"))
    print_summary(prob)
    opt = optimizer_with_attributes(
        COPT.Optimizer, "Threads" => 1, "Logging" => 0)
    model = build_sddp_model(prob; optimizer = opt)
    println("Tier2 model built; COPT Threads=1; SDDP threads=",
            Threads.nthreads(), "; out=", out)

    t0 = time()
    last_save = Ref(t0)
    function autosave(result)
        if time() - last_save[] >= 600.0
            SDDP.write_cuts_to_file(model, cuts_path)
            last_save[] = time()
            @printf("autosave cuts t=%.1fs bound=%.6f\n",
                    time() - t0, result.bound)
        end
        return nothing
    end

    SDDP.train(
        model;
        iteration_limit = 1_000_000_000,
        time_limit = train_seconds,
        duality_handler = SDDP.BanditDuality(),
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
    println(@sprintf("train done wall=%.1fs LB=%.6f", train_wall, lb))

    sims = SDDP.simulate(model, sim_paths; parallel_scheme = SDDP.Threaded())
    costs = [sum(stage[:stage_objective] for stage in path) for path in sims]
    mu = mean(costs)
    sd = std(costs)
    ci = 1.959963984540054 * sd / sqrt(length(costs))
    gap = (mu - lb) / abs(mu) * 100.0
    println(@sprintf("%d paths UB=%.6f ± %.6f gap=%.6f%%",
                     sim_paths, mu, ci, gap))
    open(summary_path, "w") do io
        println(io, "模型: Tier2 全整数等级模型 (Julia SDDP.jl, COPT)")
        println(io, "训练时限 s: ", train_seconds)
        println(io, "训练墙钟 s: ", train_wall)
        println(io, "SDDP 线程: ", Threads.nthreads())
        println(io, "求解器: COPT.Optimizer Threads=1")
        println(io, "duality: BanditDuality (Continuous + Strengthened)")
        println(io, "割类型: SINGLE_CUT")
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
