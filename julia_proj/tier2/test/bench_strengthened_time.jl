# bench_strengthened_time.jl — measure SDDP.jl StrengthenedConicDuality time
# on the same Tier2 state/scene cases used by the Python lift benchmark.
using Dates
using JSON
using JuMP
using SDDP
using COPT

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "code", "problem_data.jl"))
include(joinpath(ROOT, "code", "model.jl"))

function state_dict(prob, x)
    Ly = StateLayout(prob.N_R, prob.G, prob.L)
    d = Dict{Symbol,Float64}()
    for r in 1:prob.N_R
        d[Symbol("I[$r]")] = x[Ly.off_I + r]
    end
    for l in 1:(prob.L - 1), r in 1:prob.N_R
        d[Symbol("o[$l,$r]")] = x[Ly.off_o + (l - 1) * prob.N_R + r]
    end
    for r in 1:prob.N_R, i in 1:prob.G
        d[Symbol("n[$r,$i]")] = x[Ly.off_n + (r - 1) * prob.G + i]
    end
    for r in 1:prob.N_R
        d[Symbol("q[$r]")] = x[Ly.off_q + r]
    end
    return d
end

function main()
    prob = DefineProblem(joinpath(ROOT, "config", "problem.toml"))
    opt = optimizer_with_attributes(
        COPT.Optimizer, "Threads" => 1, "Logging" => 0, "TimeLimit" => 5.0)
    model = build_sddp_model(prob; optimizer = opt)
    SDDP._initialize_solver(model; throw_error = true)
    cases = JSON.parsefile(joinpath(ROOT, "test", "_lift_bench_states.json"))
    println("stage scene strengthen_s ok")
    for case in cases
        t = Int(case["stage"])
        s = Int(case["scene"]) + 1
        x = Float64.(case["state"])
        str_times = Float64[]
        oks = Bool[]
        for rep in 1:3
            node = model[t + 1]
            SDDP.set_incoming_state(node, state_dict(prob, x))
            SDDP.parameterize(node, prob.pool[s])
            ts = try
                @elapsed SDDP.get_dual_solution(node, SDDP.StrengthenedConicDuality())
            catch err
                println("  ERROR t=$t s=$s: ", sprint(showerror, err))
                NaN
            end
            push!(str_times, ts)
            push!(oks, isfinite(ts))
        end
        println("$t $s $(round(median(str_times), digits=4)) $oks")
    end
    println("JL_BENCH_DONE")
end

main()
