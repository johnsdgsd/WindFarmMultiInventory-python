# validate_equivalence.jl — structural + one-stage numerical equivalence check
# against the Python Tier2 StageSubproblem.
#
# Run:
#   python test/compare_python_stage.py       # writes test/_python_stage.json
#   julia  test/validate_equivalence.jl
#
# The Julia graph is built with T=2/pool=4 for speed; node 2 is the first real
# period, whose subproblem is identical to the T=60 node 2. The scene is the
# first exported pool scene (scene index 0), the same one Python uses. Incoming
# state values are taken from the Python JSON, so both sides solve the same
# fixed integer state with theta = 0. Default optimizer is COPT (same as the
# Python reference). Section C separately compares incoming_bounds=true/false
# to show the 80 statein_* rows do not change the one-stage feasible set.

const _ROOT = dirname(dirname(abspath(@__FILE__)))
include(joinpath(_ROOT, "code", "problem_data.jl"))
include(joinpath(_ROOT, "code", "model.jl"))
using SDDP, JuMP, HiGHS, COPT
import MathOptInterface as MOI

const PY_JSON = joinpath(_ROOT, "test", "_python_stage.json")
isfile(PY_JSON) || error("先运行 test/compare_python_stage.py 生成 $PY_JSON")
py = JSON.parsefile(PY_JSON)

prob = DefineProblem(joinpath(_ROOT, "config", "problem.toml"); T = 2, pool_size = 4)
model = build_sddp_model(prob)
node = model[2]
sp = node.subproblem
# A freshly built non-direct SDDP subproblem has no optimizer attached; SDDP
# attaches it during train/backward. For this standalone one-stage solve,
# attach COPT explicitly (same solver as the Python reference).
JuMP.set_optimizer(sp, COPT.Optimizer)
JuMP.set_silent(sp)
Ly = state_layout(prob)

state_sym_I(r) = Symbol("I[$(r)]")
state_sym_o(l, r) = Symbol("o[$(l),$(r)]")
state_sym_n(r, i) = Symbol("n[$(r),$(i)]")
state_sym_q(r) = Symbol("q[$(r)]")

function fix_state!(node, prob, x)
    Ly = state_layout(prob)
    for r in 1:prob.N_R
        JuMP.fix(node.states[state_sym_I(r)].in, x[ix_I(Ly, r)]; force = true)
        JuMP.fix(node.states[state_sym_q(r)].in, x[ix_q(Ly, r)]; force = true)
        for l in 1:(prob.L - 1)
            JuMP.fix(node.states[state_sym_o(l, r)].in, x[ix_o(Ly, l, r)];
                     force = true)
        end
        for i in 1:prob.G
            JuMP.fix(node.states[state_sym_n(r, i)].in, x[ix_n(Ly, r, i)];
                     force = true)
        end
    end
    return nothing
end

function julia_values(node, prob)
    sp = node.subproblem
    N_R, G, L = prob.N_R, prob.G, prob.L
    q_out = [JuMP.value(node.states[state_sym_q(r)].out) for r in 1:N_R]
    I_out = [JuMP.value(node.states[state_sym_I(r)].out) for r in 1:N_R]
    n_out = [JuMP.value(node.states[state_sym_n(r, i)].out)
             for r in 1:N_R, i in 1:G]
    taup = [JuMP.value(JuMP.variable_by_name(sp, "taup[$(r)]")) for r in 1:N_R]
    taum = [JuMP.value(JuMP.variable_by_name(sp, "taum[$(r)]")) for r in 1:N_R]
    y = [JuMP.value(JuMP.variable_by_name(sp, "y[$(r),$(m)]"))
         for r in 1:N_R, m in 1:N_R]
    S_end = [JuMP.value(JuMP.variable_by_name(sp, "S_end[$(r)]"))
             for r in 1:N_R]
    N = zeros(Float64, N_R, G, G + 1)
    for r in 1:N_R, i in 1:G, j in i:(G + 1)
        N[r, i, j] = JuMP.value(JuMP.variable_by_name(sp, "N_$(r)_$(i)_$(j)"))
    end
    F_new = [sum(N[r, i, G + 1] for i in 1:G) for r in 1:N_R]
    return (q_out = q_out, I_out = I_out, n_out = n_out, taup = taup,
            taum = taum, y = y, S_end = S_end, N = N, F_new = F_new)
end

function family(name::AbstractString)
    for pref in ("Nsum_", "Tail_", "grade0_", "grade_", "dem_", "fleet_",
                 "send_def_", "tau_bal", "tau_lo_", "tau_hi_", "tau_src_",
                 "inv_", "pipe_last_", "pipe_", "cap_", "pas_", "statein_")
        startswith(name, pref) && return rstrip(pref, '_')
    end
    return name
end

function julia_rows_by_family(sp)
    counts = Dict{String,Int}()
    for (F, S) in JuMP.list_of_constraint_types(sp)
        F == JuMP.VariableRef && continue
        for con in JuMP.all_constraints(sp, F, S)
            f = family(JuMP.name(con))
            counts[f] = get(counts, f, 0) + 1
        end
    end
    return counts
end

function julia_cols_by_family(sp)
    names = JuMP.name.(JuMP.all_variables(sp))
    c = Dict{String,Int}()
    for n in names
        key = if endswith(n, "_in")
            "state_in"
        elseif endswith(n, "_out")
            "state_out"
        elseif startswith(n, "taup[")
            "taup"
        elseif startswith(n, "taum[")
            "taum"
        elseif startswith(n, "y[")
            "y"
        elseif startswith(n, "S_end[")
            "S_end"
        elseif startswith(n, "N_")
            "N"
        else
            "other"
        end
        c[key] = get(c, key, 0) + 1
    end
    # The anonymous Bellman theta is the only unmatched column.
    c["theta"] = get(c, "other", 0)
    delete!(c, "other")
    return c
end

rows_jl = julia_rows_by_family(sp)
cols_jl = julia_cols_by_family(sp)
n_rows_jl = JuMP.num_constraints(sp; count_variable_in_set_constraints = false)
n_cols_jl = JuMP.num_variables(sp)
n_int_jl = count(JuMP.is_integer, JuMP.all_variables(sp))

println("="^88)
println("A. 结构对照（Python StageSubproblem(1) vs Julia node 2；T=2/pool=4 图）")
println("="^88)
println("状态布局: I($(prob.N_R)) | o($((prob.L-1)*prob.N_R)) | n($(prob.N_R*prob.G)) | q($(prob.N_R))  => $(state_dim(prob)) 维")
println("池: 导出 Python 池前 $(length(prob.pool)) 条；对比 scene 0；pool_source=$(prob.pool_source)")
println("Python rows=$(py["cases"][1]["rows"]) cols=$(py["cases"][1]["cols"])")
println("Julia  rows=$(n_rows_jl) cols=$(n_cols_jl) integer_cols=$(n_int_jl)")
println("Julia 行数 − Python 行数 = $(n_rows_jl - py["cases"][1]["rows"])（应为 80 条 statein_ 入状态界约束）")
println("Julia 整数列 $(n_int_jl) vs Python 整数列 $(py["cases"][1]["integer_cols"])（40 个 x_in 在 Python 也声明整数；")
println("  SDDP.jl 把 Int 放在 .out，.in 每步被上一期整数 .out 固定，见 README 的差异表）")
println("Python cols_by_family: ", py["cases"][1]["cols_by_family"])
println("Julia  cols_by_family: ", cols_jl)
println("Python rows_by_family: ", py["cases"][1]["rows_by_family"])
println("Julia  rows_by_family: ", rows_jl)

# Check the structural counts.
@assert n_cols_jl == py["cases"][1]["cols"]  # 191
@assert n_rows_jl == py["cases"][1]["rows"] + 80
for (k, v) in py["cases"][1]["rows_by_family"]
    @assert get(rows_jl, k, 0) == v "行族 $k: Julia=$(get(rows_jl,k,0)) Python=$v"
end
@assert get(rows_jl, "statein", 0) == 80
for (k, v) in py["cases"][1]["cols_by_family"]
    jk = k == "xin" ? "state_in" : k == "xout" ? "state_out" :
         k == "send" ? "S_end" : k
    @assert get(cols_jl, jk, 0) == v "列族 $k: Julia=$(get(cols_jl,jk,0)) Python=$v"
end
println("结构断言全部通过。")

println()
println("="^88)
println("B. 单阶段数值对照（固定整数状态、scene 0、theta=0）")
println("="^88)
recsum(x::Number) = Float64(x)
recsum(x::AbstractArray) = sum(recsum(y) for y in x)

py_rows(x) = permutedims(reduce(hcat, [Float64.(row) for row in x]))

default_results = Dict{String,Any}()
for case in py["cases"]
    x = Float64.(case["state"])
    fix_state!(node, prob, x)
    node.parameterize(prob.pool[1])
    JuMP.optimize!(sp)
    status = JuMP.termination_status(sp)
    status == MOI.OPTIMAL || error("Julia node 2 求解失败：$status")
    obj_jl = JuMP.objective_value(sp)
    theta = SDDP.bellman_term(node.bellman_function)
    theta_val = JuMP.value(theta)
    jv = julia_values(node, prob)
    obj_py = Float64(case["stage_cost"])
    absdiff = abs(obj_jl - obj_py)
    reldiff = absdiff / max(1.0, abs(obj_py))
    println("case=$(case["name"]): Python stage_cost=$(obj_py)  Julia objective=$(obj_jl)  " *
            "|Δ|=$(absdiff)  rel=$(reldiff)  theta=$(theta_val)")
    py_q = Float64.(case["q_out"])
    py_I = Float64.(case["I_out"])
    py_nout = py_rows(case["n_out"])
    py_y = py_rows(case["y"])
    py_N = zeros(Float64, prob.N_R, prob.G, prob.G + 1)
    for r in 1:prob.N_R, i in 1:prob.G, j in 1:(prob.G + 1)
        py_N[r, i, j] = Float64(case["N"][r][i][j])
    end
    println("  q_out  py=$(py_q)  jl=$(jv.q_out)  max|Δ|=$(maximum(abs.(py_q - jv.q_out)))")
    println("  I_out  py=$(py_I)  jl=$(jv.I_out)  max|Δ|=$(maximum(abs.(py_I - jv.I_out)))")
    println("  n_out  max|Δ|=$(maximum(abs.(py_nout - jv.n_out)))")
    println("  Σtaup  py=$(sum(Float64.(case["taup"])))  jl=$(sum(jv.taup))   " *
            "Σtaum py=$(sum(Float64.(case["taum"])))  jl=$(sum(jv.taum))")
    println("  Σy     py=$(recsum(case["y"]))  jl=$(sum(jv.y))   " *
            "ΣS_end py=$(sum(Float64.(case["S_end"])))  jl=$(sum(jv.S_end))")
    println("  y      max|Δ|=$(maximum(abs.(py_y - jv.y)))")
    println("  ΣN     py=$(recsum(case["N"]))  jl=$(sum(jv.N))   " *
            "ΣF_new py=$(sum(Float64.(case["F_new"])))  jl=$(sum(jv.F_new))")
    println("  N      max|Δ|=$(maximum(abs.(py_N - jv.N)))")
    reldiff <= 1e-6 || error("case $(case["name"]) 目标值不等价：rel=$reldiff")

    # LP relaxation (the Tier2 backward pass). Compare objective and the
    # state gradient reduced-costs; duals can differ under degeneracy.
    undo = JuMP.relax_integrality(sp)
    JuMP.optimize!(sp)
    lp_obj_jl = JuMP.objective_value(sp)
    grad_jl = zeros(Float64, Ly.dim)
    for r in 1:prob.N_R
        grad_jl[ix_I(Ly, r)] =
            JuMP.dual(JuMP.FixRef(node.states[state_sym_I(r)].in))
        grad_jl[ix_q(Ly, r)] =
            JuMP.dual(JuMP.FixRef(node.states[state_sym_q(r)].in))
        for l in 1:(prob.L - 1)
            grad_jl[ix_o(Ly, l, r)] =
                JuMP.dual(JuMP.FixRef(node.states[state_sym_o(l, r)].in))
        end
        for i in 1:prob.G
            grad_jl[ix_n(Ly, r, i)] =
                JuMP.dual(JuMP.FixRef(node.states[state_sym_n(r, i)].in))
        end
    end
    undo()
    lp_obj_py = Float64(case["lp_objective"])
    lp_grad_py = Float64.(case["lp_gradient"])
    glp = maximum(abs.(grad_jl .- lp_grad_py))
    glp_flip = maximum(abs.(grad_jl .+ lp_grad_py))
    println("  LP     py=$(lp_obj_py)  jl=$(lp_obj_jl)  " *
            "max|Δgrad|=$(glp)  max|Δgrad 反号|=$(glp_flip)")
    abs(lp_obj_jl - lp_obj_py) / max(1.0, abs(lp_obj_py)) <= 1e-6 ||
        error("case $(case["name"]) LP 松弛目标值不等价")
    default_results[case["name"]] =
        (obj = obj_jl, jv = jv, lp = lp_obj_jl, grad = copy(grad_jl))
end
println("目标值 + LP 松弛等价断言全部通过。")
println("注：LP 对偶在活跃界/零点可能不唯一；不同合法次梯度不影响割有效性。")

println()
println("="^88)
println("C. 80 条 statein_* 不改变单阶段可行域（COPT，incoming_bounds=true vs false）")
println("="^88)
model_nb = build_sddp_model(prob; optimizer = COPT.Optimizer,
                            incoming_bounds = false)
node_nb = model_nb[2]
sp_nb = node_nb.subproblem
JuMP.set_optimizer(sp_nb, COPT.Optimizer)
JuMP.set_silent(sp_nb)
n_rows_nb = JuMP.num_constraints(sp_nb; count_variable_in_set_constraints = false)
println("rows: incoming_bounds=true=$(n_rows_jl), false=$(n_rows_nb), " *
        "Python=$(py["cases"][1]["rows"])")
@assert n_rows_nb == py["cases"][1]["rows"]
for case in py["cases"]
    x = Float64.(case["state"])
    fix_state!(node_nb, prob, x)
    node_nb.parameterize(prob.pool[1])
    JuMP.optimize!(sp_nb)
    status = JuMP.termination_status(sp_nb)
    status == MOI.OPTIMAL || error("incoming_bounds=false MIP 失败：$status")
    obj_nb = JuMP.objective_value(sp_nb)
    jv_nb = julia_values(node_nb, prob)
    dr = default_results[case["name"]]
    jv_def = dr.jv
    dq = maximum(abs.(jv_nb.q_out .- jv_def.q_out))
    dI = maximum(abs.(jv_nb.I_out .- jv_def.I_out))
    dn = maximum(abs.(jv_nb.n_out .- jv_def.n_out))
    dy = maximum(abs.(jv_nb.y .- jv_def.y))
    dN = maximum(abs.(jv_nb.N .- jv_def.N))
    dF = maximum(abs.(jv_nb.F_new .- jv_def.F_new))
    println("case=$(case["name"]): obj true=$(dr.obj), false=$(obj_nb), " *
            "Python=$(case["stage_cost"]); max|Δ| q=$(dq) I=$(dI) n=$(dn) " *
            "y=$(dy) N=$(dN) F=$(dF)")
    abs(obj_nb - dr.obj) <= 1e-6 * max(1.0, abs(dr.obj)) ||
        error("case $(case["name"]) incoming_bounds=true/false 目标不等")
    maximum((dq, dI, dn, dy, dN, dF)) <= 1e-6 ||
        error("case $(case["name"]) incoming_bounds=true/false 变量不等")
    undo = JuMP.relax_integrality(sp_nb)
    JuMP.optimize!(sp_nb)
    lp_nb = JuMP.objective_value(sp_nb)
    undo()
    println("  LP: true=$(dr.lp), false=$(lp_nb), Python=$(case["lp_objective"])")
    abs(lp_nb - dr.lp) <= 1e-6 * max(1.0, abs(dr.lp)) ||
        error("case $(case["name"]) incoming_bounds=true/false LP 目标不等")
end
println("incoming_bounds=true/false 的单阶段 MIP/LP 目标与全部变量一致；")
println("80 条 statein_* 在固定 .in 处不改变可行域，只是 Python x_in 界/整数性的")
println("SDDP.jl 表达；incoming_bounds=false 的行数 146 与 Python 完全一致。")
