# validate_equivalence_extra.jl — extended independent equivalence check.
#
# Complements validate_equivalence.jl with:
#   * 6 fixed integer states x 3 scenes (0, 1, 7), MIP and LP;
#   * full row/column bound comparison against the Python fixture;
#   * explicit inspection of the 80 statein_* rows and the .in/.out split;
#   * incoming_bounds=true vs false over every case;
#   * one intentionally infeasible state (both models must agree);
#   * fixed .in after a real SDDP.train iteration.
#
# Run (Python fixture first):
#   "D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/compare_python_stage_extra.py
#   julia test/validate_equivalence_extra.jl

const _ROOT = dirname(dirname(abspath(@__FILE__)))
include(joinpath(_ROOT, "code", "problem_data.jl"))
include(joinpath(_ROOT, "code", "model.jl"))
using SDDP, JuMP, HiGHS, COPT
import MathOptInterface as MOI

const PY_JSON = joinpath(_ROOT, "test", "_python_stage_extra.json")
isfile(PY_JSON) || error("先运行 test/compare_python_stage_extra.py")
py = JSON.parsefile(PY_JSON)

let
prob = DefineProblem(joinpath(_ROOT, "config", "problem.toml"); T = 2, pool_size = 8)
model = build_sddp_model(prob)
node = model[2]
sp = node.subproblem
JuMP.set_optimizer(sp, COPT.Optimizer)
JuMP.set_silent(sp)
Ly = state_layout(prob)
N_R, G, L = prob.N_R, prob.G, prob.L

state_sym_I(r) = Symbol("I[$(r)]")
state_sym_o(l, r) = Symbol("o[$(l),$(r)]")
state_sym_n(r, i) = Symbol("n[$(r),$(i)]")
state_sym_q(r) = Symbol("q[$(r)]")

function fix_state!(nd, prob, x)
    ly = state_layout(prob)
    for r in 1:prob.N_R
        JuMP.fix(nd.states[state_sym_I(r)].in, x[ix_I(ly, r)]; force = true)
        JuMP.fix(nd.states[state_sym_q(r)].in, x[ix_q(ly, r)]; force = true)
        for l in 1:(prob.L - 1)
            JuMP.fix(nd.states[state_sym_o(l, r)].in, x[ix_o(ly, l, r)]; force = true)
        end
        for i in 1:prob.G
            JuMP.fix(nd.states[state_sym_n(r, i)].in, x[ix_n(ly, r, i)]; force = true)
        end
    end
    return nothing
end

"""State vector in the Python order (40 entries)."""
function out_vector(nd, prob)
    ly = state_layout(prob)
    x = zeros(Float64, ly.dim)
    for r in 1:prob.N_R
        x[ix_I(ly, r)] = JuMP.value(nd.states[state_sym_I(r)].out)
        x[ix_q(ly, r)] = JuMP.value(nd.states[state_sym_q(r)].out)
        for l in 1:(prob.L - 1)
            x[ix_o(ly, l, r)] = JuMP.value(nd.states[state_sym_o(l, r)].out)
        end
        for i in 1:prob.G
            x[ix_n(ly, r, i)] = JuMP.value(nd.states[state_sym_n(r, i)].out)
        end
    end
    return x
end

function controls(nd, prob, sp_)
    n_r, g = prob.N_R, prob.G
    q_out = [JuMP.value(nd.states[state_sym_q(r)].out) for r in 1:n_r]
    taup = [JuMP.value(JuMP.variable_by_name(sp_, "taup[$(r)]")) for r in 1:n_r]
    taum = [JuMP.value(JuMP.variable_by_name(sp_, "taum[$(r)]")) for r in 1:n_r]
    y = [JuMP.value(JuMP.variable_by_name(sp_, "y[$(r),$(m)]"))
         for r in 1:n_r, m in 1:n_r]
    S_end = [JuMP.value(JuMP.variable_by_name(sp_, "S_end[$(r)]")) for r in 1:n_r]
    N = zeros(Float64, n_r, g, g + 1)
    for r in 1:n_r, i in 1:g, j in i:(g + 1)
        N[r, i, j] = JuMP.value(JuMP.variable_by_name(sp_, "N_$(r)_$(i)_$(j)"))
    end
    F = [sum(N[r, i, g + 1] for i in 1:g) for r in 1:n_r]
    return (q_out = q_out, taup = taup, taum = taum, y = y, S_end = S_end,
            N = N, F = F)
end

function family(name::AbstractString)
    for pref in ("Nsum_", "Tail_", "Ndef_", "grade0_", "grade_", "dem_",
                 "fleet_", "send_def_", "tau_bal", "tau_lo_", "tau_hi_",
                 "tau_src_", "inv_", "pipe_last_", "pipe_", "cap_", "pas_",
                 "statein")
        startswith(name, pref) && return rstrip(pref, '_')
    end
    return name
end

function set_bounds(con)
    set = JuMP.constraint_object(con).set
    if set isa MOI.LessThan{Float64}
        return (-Inf, set.upper)
    elseif set isa MOI.GreaterThan{Float64}
        return (set.lower, Inf)
    elseif set isa MOI.EqualTo{Float64}
        return (set.value, set.value)
    elseif set isa MOI.Interval{Float64}
        return (set.lower, set.upper)
    end
    error("unexpected constraint set $(typeof(set))")
end

function julia_rows(sp_)
    d = Dict{String,Vector{Tuple{Float64,Float64}}}()
    for (F, S) in JuMP.list_of_constraint_types(sp_)
        F == JuMP.VariableRef && continue
        for con in JuMP.all_constraints(sp_, F, S)
            f = family(JuMP.name(con))
            push!(get!(d, f, Tuple{Float64,Float64}[]), set_bounds(con))
        end
    end
    for k in keys(d)
        sort!(d[k])
    end
    return d
end

fl(x) = Float64(x)

function compare_rows(tag, jl, expd; skip = String[])
    ok = true
    for (k, v) in expd
        k in skip && continue
        jv = get(jl, k, nothing)
        if jv === nothing
            println("  [FAIL] $(tag) row family $(k): Julia missing"); ok = false
        elseif length(jv) != length(v)
            println("  [FAIL] $(tag) row family $(k): Julia $(length(jv)) rows, Python $(length(v))"); ok = false
        elseif maximum(maximum(abs.(collect(a) .- collect(b))) for (a, b) in zip(jv, v)) > 1e-12
            println("  [FAIL] $(tag) row family $(k): bounds differ"); ok = false
        end
    end
    return ok
end

println("="^92)
println("A. 结构：列/行族、界、整数性（Python fixture vs Julia node 2）")
println("="^92)
rows_jl = julia_rows(sp)
cols = JuMP.all_variables(sp)
n_int = count(JuMP.is_integer, cols)
n_cols = length(cols)
n_rows = JuMP.num_constraints(sp; count_variable_in_set_constraints = false)
println("Julia rows=$(n_rows) cols=$(n_cols) integer_cols=$(n_int)；" *
        "Python rows=$(py["rows"]) cols=$(py["cols"])")

# Python row bound dump grouped by family.
py_rows = Dict{String,Vector{Tuple{Float64,Float64}}}()
for r in py["row_info"]
    lb = fl(r["lb"]) <= -1e29 ? -Inf : fl(r["lb"])
    ub = fl(r["ub"]) >= 1e29 ? Inf : fl(r["ub"])
    push!(get!(py_rows, r["family"], Tuple{Float64,Float64}[]), (lb, ub))
end
for k in keys(py_rows); sort!(py_rows[k]); end

@assert n_cols == py["cols"] == 191
@assert n_rows == py["rows"] + 80 == 226
ok_rows = compare_rows("A", rows_jl, py_rows; skip = ["statein"])
@assert length(get(rows_jl, "statein", [])) == 80
@assert !haskey(rows_jl, "Ndef") && !haskey(py_rows, "Ndef")
println("行族/界对照: ", ok_rows ? "PASS" : "FAIL",
        "（Python 146 行 + 80 statein = 226；Ndef 0）")

# Column bounds: states .out vs Python xout; controls vs Python controls.
println("\n列界：")
ok_bounds = true
xb = Tuple{Float64,Float64,Int}[]
for r in 1:N_R
    v = node.states[state_sym_I(r)].out
    push!(xb, (JuMP.lower_bound(v), JuMP.upper_bound(v), Int(JuMP.is_integer(v))))
    @assert JuMP.upper_bound(v) == prob.Imax[r]
    @assert JuMP.is_integer(v)
    v = node.states[state_sym_q(r)].out
    push!(xb, (JuMP.lower_bound(v), JuMP.upper_bound(v), Int(JuMP.is_integer(v))))
    @assert JuMP.upper_bound(v) == prob.Qmax[r]
    for l in 1:(L - 1)
        v = node.states[state_sym_o(l, r)].out
        push!(xb, (JuMP.lower_bound(v), JuMP.upper_bound(v), Int(JuMP.is_integer(v))))
        @assert JuMP.upper_bound(v) == prob.Qmax[r]
    end
    for i in 1:G
        v = node.states[state_sym_n(r, i)].out
        push!(xb, (JuMP.lower_bound(v), JuMP.upper_bound(v), Int(JuMP.is_integer(v))))
        @assert JuMP.upper_bound(v) == prob.N[r]
    end
end
py_xout = sort!(Tuple{Float64,Float64,Int}[(fl(c["lb"]), fl(c["ub"]),
                  c["type"] == "I" ? 1 : 0) for c in py["col_info"]
                  if c["family"] == "xout"])
@assert sort!(xb) == py_xout
println("  state .out (40) = Python xout 界/整数性：PASS")

ctrl_bounds = Tuple{Float64,Float64,Int}[]
for r in 1:N_R
    for nm in ("taup[$(r)]", "taum[$(r)]", "S_end[$(r)]")
        v = JuMP.variable_by_name(sp, nm)
        push!(ctrl_bounds, (JuMP.lower_bound(v), JuMP.upper_bound(v),
                            Int(JuMP.is_integer(v))))
    end
    for m in 1:N_R
        v = JuMP.variable_by_name(sp, "y[$(r),$(m)]")
        push!(ctrl_bounds, (JuMP.lower_bound(v), JuMP.upper_bound(v),
                            Int(JuMP.is_integer(v))))
    end
    for i in 1:G, j in i:(G + 1)
        v = JuMP.variable_by_name(sp, "N_$(r)_$(i)_$(j)")
        push!(ctrl_bounds, (JuMP.lower_bound(v), JuMP.upper_bound(v),
                            Int(JuMP.is_integer(v))))
    end
end
py_ctrl = sort!(Tuple{Float64,Float64,Int}[(fl(c["lb"]), fl(c["ub"]),
                 c["type"] == "I" ? 1 : 0) for c in py["col_info"]
                 if c["family"] in ("taup", "taum", "y", "send", "N")])
@assert sort!(ctrl_bounds) == py_ctrl
println("  控制列 (5 taup + 5 taum + 25 y + 5 send + 70 N) = Python 界/整数性：PASS")
@assert n_int == 40 + 110
println("  整数列 $(n_int) = 40 .out + 110 控制；Python 190 = 这 150 + 40 个 x_in（固定后是常数）")

# The remaining continuous columns must be exactly the 40 .in and theta.
in_vars = JuMP.VariableRef[]
for r in 1:N_R
    push!(in_vars, node.states[state_sym_I(r)].in)
    push!(in_vars, node.states[state_sym_q(r)].in)
    for l in 1:(L - 1)
        push!(in_vars, node.states[state_sym_o(l, r)].in)
    end
    for i in 1:G
        push!(in_vars, node.states[state_sym_n(r, i)].in)
    end
end
@assert length(in_vars) == 40
@assert all(!JuMP.is_integer(v) for v in in_vars)
@assert all(!JuMP.has_lower_bound(v) for v in in_vars)
cont_vars = [v for v in cols if !JuMP.is_integer(v)]
@assert length(cont_vars) == 41
println("  连续列 41 = 40 个 .in（SDDP 每步 JuMP.fix，模型里无变量界）+ 1 个 Bellman theta")

# ---- statein_* rows encode exactly the Python x_in bounds ----
println("\n80 条 statein_* 行：")
in_index = Dict{JuMP.VariableRef,Int}()
for r in 1:N_R
    in_index[node.states[state_sym_I(r)].in] = ix_I(Ly, r)
    in_index[node.states[state_sym_q(r)].in] = ix_q(Ly, r)
    for l in 1:(L - 1)
        in_index[node.states[state_sym_o(l, r)].in] = ix_o(Ly, l, r)
    end
    for i in 1:G
        in_index[node.states[state_sym_n(r, i)].in] = ix_n(Ly, r, i)
    end
end
seen = Dict{Int,Int}()
n_statein = 0
for (F, S) in JuMP.list_of_constraint_types(sp)
    F == JuMP.VariableRef && continue
    for con in JuMP.all_constraints(sp, F, S)
        startswith(JuMP.name(con), "statein") || continue
        n_statein += 1
        obj = JuMP.constraint_object(con)
        @assert obj.func isa JuMP.AffExpr && length(obj.func.terms) == 1
        v, coeff = first(obj.func.terms)
        @assert coeff == 1.0
        j = in_index[v]
        ub = fl(py["state_ub"][j])
        if obj.set isa MOI.GreaterThan{Float64}
            @assert obj.set.lower == 0.0
        elseif obj.set isa MOI.LessThan{Float64}
            @assert obj.set.upper == ub
        else
            error("unexpected statein set $(typeof(obj.set))")
        end
        seen[j] = get(seen, j, 0) + 1
    end
end
@assert n_statein == 80
@assert length(seen) == 40 && all(v == 2 for v in values(seen))
println("  恰好 80 行，覆盖 40 个 .in × {lo 0, hi Python x_in ub}；PASS")

println("\n" * "="^92)
println("B. 场景系数：Tail 的 n.in 系数 = -tail_cdf(scene)，Nsum = -1")
println("="^92)
for si in py["scenes"]
    scene = prob.pool[Int(si) + 1]
    c = tail_cdf(scene[2])
    node.parameterize(scene)
    for r in 1:N_R, i in 1:G, j in (i + 1):(G + 1)
        con = JuMP.constraint_by_name(sp, "Tail_$(r)_$(i)_$(j)")
        n_in = node.states[state_sym_n(r, i)].in
        @assert isapprox(JuMP.normalized_coefficient(con, n_in), -c[r, i, j];
                         atol = 1e-12)
        @assert set_bounds(con) == (-0.5, 0.5)
    end
    for r in 1:N_R, i in 1:G
        con = JuMP.constraint_by_name(sp, "Nsum_$(r)_$(i)")
        n_in = node.states[state_sym_n(r, i)].in
        @assert JuMP.normalized_coefficient(con, n_in) == -1.0
        @assert set_bounds(con) == (0.0, 0.0)
    end
    println("  scene $(si)：50 Tail + 20 Nsum 系数与 tail_cdf 一致；PASS")
end

println("\n" * "="^92)
println("C. 单阶段数值对照：$(length(py["cases"])) 个 (状态,场景) 组合，MIP + LP")
println("="^92)
max_obj = 0.0
max_x = 0.0
max_ctrl = 0.0
max_N = 0.0
max_lp = 0.0
max_grad = 0.0
n_cases = 0
all_int = true
all_in_fixed = true
grad_records = Dict{String,Any}[]
for case in py["cases"]
    si = Int(case["scene_index"])
    x = fl.(case["state"])
    fix_state!(node, prob, x)
    node.parameterize(prob.pool[si + 1])
    JuMP.optimize!(sp)
    status = JuMP.termination_status(sp)
    status == MOI.OPTIMAL || error("case $(case["name"])/scene $(si): $status")
    theta = JuMP.value(SDDP.bellman_term(node.bellman_function))
    obj = JuMP.objective_value(sp)
    xout = out_vector(node, prob)
    cv = controls(node, prob, sp)
    d_obj = maximum((abs(obj - fl(case["objective"])),
                     abs((obj - theta) - fl(case["stage_cost"]))))
    max_obj = max(max_obj, d_obj)
    d_x = maximum(abs.(xout .- fl.(case["x_out"])))
    max_x = max(max_x, d_x)
    py_y = fl.(reduce(hcat, [Float64.(row) for row in case["y"]]))
    py_N = zeros(Float64, N_R, G, G + 1)
    for r in 1:N_R, i in 1:G, j in (i):(G + 1)
        py_N[r, i, j] = fl(case["N"][r][i][j])
    end
    d_ctrl = maximum((maximum(abs.(cv.taup .- fl.(case["taup"]))),
                      maximum(abs.(cv.taum .- fl.(case["taum"]))),
                      maximum(abs.(cv.y .- py_y)),
                      maximum(abs.(cv.S_end .- fl.(case["S_end"])))))
    max_ctrl = max(max_ctrl, d_ctrl)
    d_N = maximum(abs.(cv.N .- py_N))
    max_N = max(max_N, d_N)
    vals = vcat(xout, cv.taup, cv.taum, cv.y[:], cv.S_end, cv.N[:])
    int_ok = all(abs.(vals .- round.(vals)) .<= 1e-6)
    all_int &= int_ok
    # Every .in must be fixed to the integer input state (SDDP does the same).
    in_ok = true
    for (v, j) in in_index
        in_ok &= JuMP.is_fixed(v) && abs(JuMP.fix_value(v) - x[j]) <= 1e-12
    end
    all_in_fixed &= in_ok

    # LP relaxation (the Python Tier2 backward pass).
    undo = JuMP.relax_integrality(sp)
    JuMP.optimize!(sp)
    lp_obj = JuMP.objective_value(sp)
    grad = zeros(Float64, Ly.dim)
    for (v, j) in in_index
        grad[j] = JuMP.dual(JuMP.FixRef(v))
    end
    undo()
    d_lp = abs(lp_obj - fl(case["lp_objective"])) /
           max(1.0, abs(fl(case["lp_objective"])))
    max_lp = max(max_lp, d_lp)
    d_g = maximum(abs.(grad .- fl.(case["lp_gradient"])))
    max_grad = max(max_grad, d_g)
    if d_g > 1e-6
        println("    LP 梯度差异分量（LP 目标相同，检查是否退化）:")
        for j in 1:Ly.dim
            if abs(grad[j] - fl(case["lp_gradient"][j])) > 1e-6
                println("      state[$j]=$(x[j])  py=$(case["lp_gradient"][j])  " *
                        "jl=$(grad[j])")
            end
        end
    end
    push!(grad_records, Dict("name" => case["name"], "scene_index" => si,
                             "lp_objective" => lp_obj,
                             "lp_gradient" => collect(grad)))
    n_cases += 1
    println(@sprintf("  %-14s scene %d: MIP %.6f/%.6f |d|=%.2e  LP %.6f/%.6f  x|d|=%.2e ctrl|d|=%.2e N|d|=%.2e int=%s grad|d|=%.2e",
                     case["name"], si, obj, fl(case["objective"]), d_obj,
                     lp_obj, fl(case["lp_objective"]), d_x, d_ctrl, d_N,
                     int_ok, d_g))
end
open(joinpath(_ROOT, "test", "_julia_stage_extra.json"), "w") do io
    JSON.print(io, Dict("cases" => grad_records))
end
println("汇总: max MIP |d|=$(max_obj)  state|d|=$(max_x)  ctrl|d|=$(max_ctrl)  " *
        "N|d|=$(max_N)  LP rel|d|=$(max_lp)  grad|d|=$(max_grad)")
@assert max_obj <= 1e-6
@assert max_x <= 1e-6 && max_ctrl <= 1e-6 && max_N <= 1e-6
@assert max_lp <= 1e-6
@assert all_int
@assert all_in_fixed
println("MIP 目标、全部状态/控制/N、LP 松弛目标一致；全部 MIP 值整数；40 个 .in " *
        "固定为输入整数状态；PASS")

println("\n" * "="^92)
println("D. incoming_bounds=true vs false（同一 80 行的影响）")
println("="^92)
model_nb = build_sddp_model(prob; optimizer = COPT.Optimizer,
                            incoming_bounds = false)
node_nb = model_nb[2]
sp_nb = node_nb.subproblem
JuMP.set_optimizer(sp_nb, COPT.Optimizer)
JuMP.set_silent(sp_nb)
@assert JuMP.num_constraints(sp_nb; count_variable_in_set_constraints = false) == 146
max_dn_obj = 0.0
max_dn_var = 0.0
for case in py["cases"]
    si = Int(case["scene_index"])
    x = fl.(case["state"])
    fix_state!(node_nb, prob, x)
    node_nb.parameterize(prob.pool[si + 1])
    JuMP.optimize!(sp_nb)
    JuMP.termination_status(sp_nb) == MOI.OPTIMAL ||
        error("false case $(case["name"])/scene $(si): $(JuMP.termination_status(sp_nb))")
    obj_nb = JuMP.objective_value(sp_nb)
    xout_nb = out_vector(node_nb, prob)
    cv_nb = controls(node_nb, prob, sp_nb)
    d_obj = abs(obj_nb - fl(case["objective"]))
    d_var = maximum((maximum(abs.(xout_nb .- fl.(case["x_out"]))),
                     maximum(abs.(cv_nb.N .- reshape(
                         [fl(case["N"][r][i][j]) for r in 1:N_R, i in 1:G,
                          j in 1:(G + 1)], N_R, G, G + 1)))))
    max_dn_obj = max(max_dn_obj, d_obj)
    max_dn_var = max(max_dn_var, d_var)
    undo = JuMP.relax_integrality(sp_nb)
    JuMP.optimize!(sp_nb)
    lp_nb = JuMP.objective_value(sp_nb)
    undo()
    max_dn_obj = max(max_dn_obj, abs(lp_nb - fl(case["lp_objective"])) /
                     max(1.0, abs(fl(case["lp_objective"]))))
end
println("incoming_bounds=false: rows=146（=Python）；MIP/LP 目标与变量 max|d|=" *
        "$(max_dn_obj)/$(max_dn_var)")
@assert max_dn_obj <= 1e-6 && max_dn_var <= 1e-6
println("80 条 statein_* 不改变任一 (状态,场景) 的单阶段 MIP/LP 可行域与最优值；PASS")

println("\n" * "="^92)
println("E. 不可行状态：两个模型必须一致判不可行")
println("="^92)
x_inf = fl.(py["infeasible"]["state"])
fix_state!(node, prob, x_inf)
node.parameterize(prob.pool[Int(py["infeasible"]["scene_index"]) + 1])
JuMP.optimize!(sp)
st = JuMP.termination_status(sp)
println("Python status=$(py["infeasible"]["python_status"]) (INFEASIBLE=2)；" *
        "Julia status=$(st)")
@assert st == MOI.INFEASIBLE
println("不可行性一致；PASS")

println("\n" * "="^92)
println("F. 真实 SDDP.train 一轮：.in 固定、LB、数值问题")
println("="^92)
prob4 = DefineProblem(joinpath(_ROOT, "config", "problem.toml"); T = 2, pool_size = 4)
model_tr = build_sddp_model(prob4)
SDDP.train(model_tr;
    iteration_limit = 1,
    print_level = 1,
    run_numerical_stability_report = true,
    duality_handler = SDDP.ContinuousConicDuality(),
    parallel_scheme = SDDP.Serial(),
)
lb = SDDP.calculate_bound(model_tr)
n_fixed = 0
n_in_total = 0
for (_, nd) in model_tr.nodes
    for (_, st) in nd.states
        n_in_total += 1
        n_fixed += JuMP.is_fixed(st.in) ? 1 : 0
    end
end
println("LB = $(lb)；.in fixed = $(n_fixed)/$(n_in_total)")
@assert n_fixed == n_in_total
@assert isfinite(lb)
println("全部 .in 在训练后被 JuMP.fix（整数 .out 的值）；PASS")

println("\n" * "="^92)
println("EXTENDED EQUIVALENCE: PASS")
println("="^92)
end
