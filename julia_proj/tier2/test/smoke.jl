# smoke.jl — structural checks + one built-in SDDP.train iteration.
#
#   julia test/smoke.jl
#
# T=2, pool=4 (first four scenes of the exported Python Tier2 pool). This is
# intentionally not a long training run: it only proves that the graph builds,
# all 40 states / 110 controls are integer, Theta is a normalized coefficient
# on n.in, and SDDP.jl can take one forward/backward step with the default COPT
# solver (both the Tier2-matching LP-relaxation and Lagrangian handlers) and
# with the optional HiGHS cross-check.

const _ROOT = dirname(dirname(abspath(@__FILE__)))
include(joinpath(_ROOT, "code", "problem_data.jl"))
include(joinpath(_ROOT, "code", "model.jl"))
using SDDP, JuMP, HiGHS, COPT

prob = DefineProblem(joinpath(_ROOT, "config", "problem.toml"); T = 2, pool_size = 4)
print_summary(prob)
@assert prob.T == 2
@assert length(prob.pool) == 4
@assert startswith(prob.pool_source, "exported")
@assert state_dim(prob) == 40
@assert prob.n0_mode == "integer_largest_remainder"
for r in 1:prob.N_R
    @assert sum(prob.n0[r, :]) == prob.N[r]
end

model = build_sddp_model(prob)                 # default: COPT
@assert length(model.nodes) == prob.T + 1
node = model[2]
@assert node.optimizer === COPT.Optimizer      # builder default is COPT
sp = node.subproblem

# ---- state / control integrality and dimensions ----
@assert length(node.states) == 40
state_out = [node.states[k].out for k in keys(node.states)]
@assert all(JuMP.is_integer.(state_out))
@assert JuMP.num_variables(sp) == 191
@assert count(JuMP.is_integer, JuMP.all_variables(sp)) == 150
var_names = JuMP.name.(JuMP.all_variables(sp))
@assert count(n -> startswith(n, "N_"), var_names) == 70

con_names = String[]
for (F, S) in JuMP.list_of_constraint_types(sp)
    F == JuMP.VariableRef && continue
    for con in JuMP.all_constraints(sp, F, S)
        push!(con_names, JuMP.name(con))
    end
end
@assert count(n -> startswith(n, "Nsum_"), con_names) == 20
@assert count(n -> startswith(n, "Tail_"), con_names) == 50
@assert count(n -> startswith(n, "Ndef_"), con_names) == 0
@assert count(n -> startswith(n, "statein_"), con_names) == 80

# Theta is a coefficient of n.in, never a right-hand-side constant. Check all
# 50 tail rows against tail_cdf(scene 1), then restore the construction state
# (the last parameterize call uses the last pool scene).
scene1 = prob.pool[1]
c1 = tail_cdf(scene1[2])
node.parameterize(scene1)
for r in 1:prob.N_R, i in 1:prob.G, j in (i+1):(prob.G+1)
    con = JuMP.constraint_by_name(sp, "Tail_$(r)_$(i)_$(j)")
    con === nothing && error("没有找到 Tail_$(r)_$(i)_$(j)")
    n_in = node.states[Symbol("n[$(r),$(i)]")].in
    @assert isapprox(JuMP.normalized_coefficient(con, n_in), -c1[r, i, j];
                     atol = 1e-12)
end
node.parameterize(prob.pool[end])
n_in = node.states[Symbol("n[1,1]")].in
println("  Tail 全部 50 行与 tail_cdf(scene 1) 一致；")
println("  scene 1 的 C[1,1,2]=$(c1[1,1,2]), C[1,1,$(prob.G)]=$(c1[1,1,prob.G]), " *
        "C[1,1,$(prob.G+1)]=$(c1[1,1,prob.G+1])")
sum11 = JuMP.constraint_by_name(sp, "Nsum_1_1")
sum11 === nothing && error("没有找到 Nsum_1_1")
@assert isapprox(JuMP.normalized_coefficient(sum11, n_in), -1.0; atol = 1e-12)
println("结构自检通过：40 状态 / 150 整数自由度 / 70 个 N / 20 Nsum / 50 Tail；")


# ---- one built-in iteration, COPT, Tier2-matching LP-relaxation backward ----
println("\n--- SDDP.train(1, COPT, ContinuousConicDuality) ---")
SDDP.train(model;
    iteration_limit = 1,
    print_level = 1,
    run_numerical_stability_report = false,
    duality_handler = SDDP.ContinuousConicDuality(),
    parallel_scheme = SDDP.Serial(),
)
lb_cont = SDDP.calculate_bound(model)
@assert isfinite(lb_cont)

# ---- one built-in iteration, COPT, optional SDDiP-style Lagrangian backward ----
println("\n--- SDDP.train(1, COPT, LagrangianDuality/BFGS(3)) ---")
model_lag = build_sddp_model(prob)
SDDP.train(model_lag;
    iteration_limit = 1,
    print_level = 1,
    run_numerical_stability_report = false,
    duality_handler = SDDP.LagrangianDuality(
        method = SDDP.LocalImprovementSearch.BFGS(3)),
    parallel_scheme = SDDP.Serial(),
)
lb_lag = SDDP.calculate_bound(model_lag)
@assert isfinite(lb_lag)

# ---- HiGHS is optional: same model, one iteration, same handler ----
println("\n--- SDDP.train(1, HiGHS, ContinuousConicDuality) ---")
model_highs = build_sddp_model(prob; optimizer = HiGHS.Optimizer)
@assert model_highs[2].optimizer === HiGHS.Optimizer
SDDP.train(model_highs;
    iteration_limit = 1,
    print_level = 1,
    run_numerical_stability_report = false,
    duality_handler = SDDP.ContinuousConicDuality(),
    parallel_scheme = SDDP.Serial(),
)
lb_highs = SDDP.calculate_bound(model_highs)
@assert isfinite(lb_highs)

# ---- same HiGHS model, optional Lagrangian handler ----
println("\n--- SDDP.train(1, HiGHS, LagrangianDuality/BFGS(3)) ---")
model_highs_lag = build_sddp_model(prob; optimizer = HiGHS.Optimizer)
SDDP.train(model_highs_lag;
    iteration_limit = 1,
    print_level = 1,
    run_numerical_stability_report = false,
    duality_handler = SDDP.LagrangianDuality(
        method = SDDP.LocalImprovementSearch.BFGS(3)),
    parallel_scheme = SDDP.Serial(),
)
lb_highs_lag = SDDP.calculate_bound(model_highs_lag)
@assert isfinite(lb_highs_lag)

println("\nsmoke 通过：LB(COPT/continuous) = $lb_cont, " *
        "LB(COPT/lagrangian) = $lb_lag, " *
        "LB(HiGHS/continuous) = $lb_highs, " *
        "LB(HiGHS/lagrangian) = $lb_highs_lag")
