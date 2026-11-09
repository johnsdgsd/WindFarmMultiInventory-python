# compare_stage.jl — 与 Python 的单阶段 LP 逐行/逐变量等价性检查。
#
# 先运行 test/dump_stage.py 生成 _py_stage_dump.toml，再运行本脚本。
# 检查项：
#   · Python 导出的场景池与 Julia 读取结果逐值一致；
#   · 虚拟阶段/真实阶段的列数、结构行数、行名集合一致；
#   · 真实阶段全部 146 行的 lb/ub/系数逐项一致；
#   · 目标系数（共享变量）一致；
#   · 同一状态/场景下的阶段成本一致；
#   · 变量取值（q.out 等退化分量单独报告）。
#
# 注意：Python 的 x_out 会对 n 块做 state_round_digits 位 round + repair；
# Julia 的 LP 值是原始值。n_out 允许 ≤5e-7 的舍入差，其余分量应一致。

using TOML, Printf, LinearAlgebra

include(joinpath(@__DIR__, "..", "code", "problem_data.jl"))
include(joinpath(@__DIR__, "..", "code", "model.jl"))

# 与 dump_stage.py 一致：忽略低于 COPT MatrixTol 量级的 Θ 系数。
const COEFF_EPS = 1e-10
const DUMP_PATH = joinpath(@__DIR__, "_py_stage_dump.toml")

"""把状态向量（Python 布局 I|o|n|q，0-based 结构）转成 SDDP 的状态字典。"""
function incoming_state_dict(prob::ProblemData, x::AbstractVector)
    N_R, G, L = prob.N_R, prob.G, prob.L
    off_o = N_R
    off_n = off_o + (L - 1) * N_R
    off_q = off_n + N_R * G
    length(x) == off_q + N_R || error("状态维 $(length(x)) ≠ $(off_q + N_R)")
    d = Dict{Symbol,Float64}()
    for r in 1:N_R
        d[Symbol("I[$(r)]")] = x[r]
    end
    for l in 1:(L - 1), r in 1:N_R
        d[Symbol("o[$(l),$(r)]")] = x[off_o + (l - 1) * N_R + (r - 1) + 1]
    end
    for r in 1:N_R, i in 1:G
        d[Symbol("n[$(r),$(i)]")] = x[off_n + (r - 1) * G + (i - 1) + 1]
    end
    for r in 1:N_R
        d[Symbol("q[$(r)]")] = x[off_q + (r - 1) + 1]
    end
    return d
end

"""状态变量 → Python 名字（xin_k/xout_k）的映射。"""
function state_var_maps(prob::ProblemData, node)
    N_R, G, L = prob.N_R, prob.G, prob.L
    off_o = N_R
    off_n = off_o + (L - 1) * N_R
    off_q = off_n + N_R * G
    vin = Dict{JuMP.VariableRef,String}()
    vout = Dict{JuMP.VariableRef,String}()
    for (k, st) in node.states
        s = String(k)
        idx = 0
        if (m = match(r"^I\[(\d+)\]$", s)) !== nothing
            idx = parse(Int, m.captures[1]) - 1
        elseif (m = match(r"^o\[(\d+),(\d+)\]$", s)) !== nothing
            l = parse(Int, m.captures[1]); r = parse(Int, m.captures[2])
            idx = off_o + (l - 1) * N_R + (r - 1)
        elseif (m = match(r"^n\[(\d+),(\d+)\]$", s)) !== nothing
            r = parse(Int, m.captures[1]); i = parse(Int, m.captures[2])
            idx = off_n + (r - 1) * G + (i - 1)
        elseif (m = match(r"^q\[(\d+)\]$", s)) !== nothing
            idx = off_q + parse(Int, m.captures[1]) - 1
        else
            error("未知状态变量名：$s")
        end
        vin[st.in] = "xin_$(idx)"
        vout[st.out] = "xout_$(idx)"
    end
    return vin, vout
end

"""决策变量 → Python 名字的映射（变量本身已按 Python 名字命名）。"""
function decision_var_map(prob::ProblemData, sub)
    N_R, G = prob.N_R, prob.G
    d = Dict{JuMP.VariableRef,String}()
    function add(name)
        v = JuMP.variable_by_name(sub, name)
        v === nothing && error("找不到变量 $name")
        d[v] = name
    end
    for r in 1:N_R
        add("taup_$(r - 1)")
        add("taum_$(r - 1)")
        add("send_$(r - 1)")
    end
    for r in 1:N_R, m in 1:N_R
        add("y_$(r - 1)_$(m - 1)")
    end
    for r in 1:N_R, i in 1:G, j in i:(G + 1)
        add("N_$(r - 1)_$(i - 1)_$(j - 1)")
    end
    return d
end

function collect_rows(sub, varmap::Dict{JuMP.VariableRef,String})
    rows = Dict{String,NamedTuple}()
    for (F, S) in JuMP.list_of_constraint_types(sub)
        F == JuMP.VariableRef && continue
        for con in JuMP.all_constraints(sub, F, S)
            name = JuMP.name(con)
            isempty(name) && continue
            obj = JuMP.constraint_object(con)
            set = obj.set
            lb, ub = if set isa JuMP.MOI.EqualTo
                (set.value, set.value)
            elseif set isa JuMP.MOI.LessThan
                (-1e30, set.upper)
            elseif set isa JuMP.MOI.GreaterThan
                (set.lower, 1e30)
            elseif set isa JuMP.MOI.Interval
                (set.lower, set.upper)
            else
                error("未知约束集 $(typeof(set))（$(name)）")
            end
            coeffs = Dict{String,Float64}()
            f = obj.func
            if f isa JuMP.VariableRef
                py = get(varmap, f, nothing)
                py !== nothing && (coeffs[py] = 1.0)
            elseif f isa JuMP.GenericAffExpr
                for (v, c) in f.terms
                    py = get(varmap, v, nothing)
                    py === nothing && continue
                    abs(c) > COEFF_EPS && (coeffs[py] = float(c))
                end
            else
                error("未知行表达式类型 $(typeof(f))（$(name)）")
            end
            rows[name] = (; lb = float(lb), ub = float(ub), coeffs = coeffs)
        end
    end
    return rows
end

function collect_obj(sub, varmap::Dict{JuMP.VariableRef,String})
    d = Dict{String,Float64}()
    f = JuMP.objective_function(sub)
    if f isa JuMP.VariableRef
        py = get(varmap, f, nothing)
        py !== nothing && (d[py] = 1.0)
    elseif f isa JuMP.GenericAffExpr
        for (v, c) in f.terms
            py = get(varmap, v, nothing)
            py === nothing && continue
            abs(c) > COEFF_EPS && (d[py] = float(c))
        end
    elseif !(f isa Number)
        error("未知目标类型 $(typeof(f))")
    end
    return d
end

"""变量界（共享变量）；未设的界用 ±1e30 表示，与 Python dump 一致。"""
function collect_bounds(sub, varmap::Dict{JuMP.VariableRef,String})
    d = Dict{String,NTuple{2,Float64}}()
    for (v, name) in varmap
        if JuMP.is_fixed(v)
            val = JuMP.fix_value(v)
            d[name] = (float(val), float(val))
        else
            lb = JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -1e30
            ub = JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : 1e30
            d[name] = (float(lb), float(ub))
        end
    end
    return d
end

function collect_x(prob::ProblemData, node)
    vin, vout = state_var_maps(prob, node)
    n = state_dim(prob)
    xin, xout = zeros(n), zeros(n)
    for (v, name) in vin
        idx = parse(Int, split(name, "_")[2]) + 1
        xin[idx] = JuMP.value(v)
    end
    for (v, name) in vout
        idx = parse(Int, split(name, "_")[2]) + 1
        xout[idx] = JuMP.value(v)
    end
    return xin, xout
end

function collect_decisions(prob::ProblemData, sub)
    taup = [JuMP.value(JuMP.variable_by_name(sub, "taup_$(r - 1)"))
            for r in 1:prob.N_R]
    taum = [JuMP.value(JuMP.variable_by_name(sub, "taum_$(r - 1)"))
            for r in 1:prob.N_R]
    send = [JuMP.value(JuMP.variable_by_name(sub, "send_$(r - 1)"))
            for r in 1:prob.N_R]
    y = Float64[]
    for r in 1:prob.N_R, m in 1:prob.N_R
        push!(y, JuMP.value(JuMP.variable_by_name(sub, "y_$(r - 1)_$(m - 1)")))
    end
    Nvals = Float64[]
    for r in 1:prob.N_R, i in 1:prob.G, j in i:(prob.G + 1)
        push!(Nvals, JuMP.value(JuMP.variable_by_name(
            sub, "N_$(r - 1)_$(i - 1)_$(j - 1)")))
    end
    return taup, taum, send, y, Nvals
end

function stage_cost_of(prob::ProblemData, xin, xout, taup, taum, y)
    off_n = prob.N_R + (prob.L - 1) * prob.N_R
    off_q = off_n + prob.N_R * prob.G
    n_in = Float64[]
    for r in 1:prob.N_R, i in 1:prob.G
        push!(n_in, xin[off_n + (r - 1) * prob.G + i])
    end
    q_in = xin[(off_q + 1):(off_q + prob.N_R)]
    I_in = xin[1:prob.N_R]
    I_out = xout[1:prob.N_R]
    ordering = prob.C * sum(q_in)
    holding = prob.h / 2 * sum(I_in .+ I_out)
    downtime = prob.Cs * sum(prob.N) - prob.Cs * sum(n_in)
    active = prob.atc / 2 * sum(taup .+ taum)
    passive = 0.0; k = 0
    for r in 1:prob.N_R, m in 1:prob.N_R
        k += 1
        passive += prob.tc_route[r, m] * y[k]
    end
    return ordering + holding + downtime + active + passive
end

"""固定 x_in 的既约成本 = ∂Q/∂x（Benders 割斜率），按 Python 状态布局排列。"""
function collect_grad(prob::ProblemData, node)
    vin, _ = state_var_maps(prob, node)
    g = zeros(state_dim(prob))
    for (v, name) in vin
        idx = parse(Int, split(name, "_")[2]) + 1
        g[idx] = JuMP.reduced_cost(v)
    end
    return g
end

function maxdiff(a, b)
    length(a) == length(b) || error("长度 $(length(a)) ≠ $(length(b))")
    isempty(a) && return 0.0
    return maximum(abs.(Float64.(a) .- Float64.(b)))
end

function row_table(py_rows, jl_rows)
    pk, jk = Set(keys(py_rows)), Set(keys(jl_rows))
    missing_jl = sort(collect(setdiff(pk, jk)))
    extra_jl = sort(collect(setdiff(jk, pk)))
    max_lb = 0.0; max_ub = 0.0; max_c = 0.0
    key_mismatch = String[]
    for name in intersect(pk, jk)
        p, j = py_rows[name], jl_rows[name]
        max_lb = max(max_lb, abs(p["lb"] - j.lb))
        max_ub = max(max_ub, abs(p["ub"] - j.ub))
        pc = Dict{String,Float64}(String(k) => float(v) for (k, v) in p["coeffs"])
        jc = j.coeffs
        if Set(keys(pc)) != Set(keys(jc))
            push!(key_mismatch, name)
        end
        for k in intersect(Set(keys(pc)), Set(keys(jc)))
            max_c = max(max_c, abs(pc[k] - jc[k]))
        end
    end
    return (; missing_jl, extra_jl, max_lb, max_ub, max_c, key_mismatch,
            n_common = length(intersect(pk, jk)))
end

function main()
    isfile(DUMP_PATH) || error("找不到 $(DUMP_PATH)；先运行 test/dump_stage.py")
    dump = TOML.parsefile(DUMP_PATH)
    meta_path = joinpath(PROJECT_ROOT, "data", "pool_seed2026_meta.toml")
    meta = TOML.parsefile(meta_path)

    prob = DefineProblem()
    model = build_sddp_model(prob)

    println("="^78)
    println("1) 场景池一致性（Python 导出文件 → Julia 读取）")
    println("="^78)
    z0, th0 = prob.pool[1]
    z0c = Float64[]; for i in 1:3, r in 1:prob.N_R; push!(z0c, z0[i, r]); end
    th0c = Float64[]
    for r in 1:prob.N_R, i in 1:prob.G, j in 1:(prob.G + 1)
        push!(th0c, th0[r, i, j])
    end
    thbar = zeros(Float64, prob.N_R, prob.G, prob.G + 1)
    for (_, th) in prob.pool; thbar .+= th; end
    thbar ./= length(prob.pool)
    thbarc = Float64[]
    for r in 1:prob.N_R, i in 1:prob.G, j in 1:(prob.G + 1)
        push!(thbarc, thbar[r, i, j])
    end
    pref_c = Float64[]
    for i in 1:prob.G, j in 1:(prob.G + 1); push!(pref_c, prob.P_ref[i, j]); end
    f_pool = mean(_cov_factor(prob.beta_cov, prob.z_bar, @view z[:, r])
                  for (z, _) in prob.pool for r in 1:prob.N_R)
    d_z = maxdiff(z0c, meta["first_z"])
    d_th = maxdiff(th0c, meta["first_theta"])
    d_bar = maxdiff(thbarc, meta["mean_theta"])
    d_pref = maxdiff(pref_c, meta["P_ref"])
    d_f = abs(f_pool - meta["pool_mean_f"])
    @printf("first_z max|Δ|     = %.3e\n", d_z)
    @printf("first_theta max|Δ| = %.3e\n", d_th)
    @printf("mean_theta max|Δ|  = %.3e\n", d_bar)
    @printf("P_ref max|Δ|       = %.3e\n", d_pref)
    @printf("pool_mean_f Δ      = %.3e\n", d_f)
    (d_z <= 1e-12 && d_th <= 1e-12 && d_bar <= 1e-12 &&
     d_pref <= 1e-12 && d_f <= 1e-12) || error("场景池与 Python 导出不一致")

    # ---- 虚拟阶段结构 ----
    node1 = model.nodes[1]
    vin1, vout1 = state_var_maps(prob, node1)
    rows1 = collect_rows(node1.subproblem, merge(vin1, vout1))
    println()
    println("="^78)
    println("2) 结构对比")
    println("="^78)
    @printf("%-28s %10s %10s\n", "item", "python", "julia")
    @printf("%-28s %10d %10d\n", "virtual columns",
            dump["virtual_n_cols"], JuMP.num_variables(node1.subproblem))
    @printf("%-28s %10d %10d\n", "virtual structural rows",
            dump["virtual_n_rows"], length(rows1))
    virt_names_ok = sort(collect(keys(rows1))) == sort(String.(dump["virtual_rows"]))
    @printf("%-28s %10s %10s\n", "virtual row names", "ok", virt_names_ok ? "ok" : "DIFF")

    # ---- 真实阶段：固定状态 + 场景 ----
    node = model.nodes[2]
    x_in_py = Float64.(dump["x_in"])
    SDDP.set_incoming_state(node, incoming_state_dict(prob, x_in_py))
    SDDP.parameterize(node, prob.pool[Int(dump["scene"]) + 1])
    # SDDP 训练前子问题没有挂求解器；这里手动挂默认求解器（COPT）做单次 LP。
    JuMP.set_optimizer(node.subproblem, default_optimizer())
    JuMP.set_silent(node.subproblem)
    SDDP.set_objective(node)
    JuMP.optimize!(node.subproblem)
    JuMP.primal_status(node.subproblem) == JuMP.MOI.FEASIBLE_POINT ||
        error("Julia 子问题未得到可行解：$(JuMP.termination_status(node.subproblem))")

    vin, vout = state_var_maps(prob, node)
    dmap = decision_var_map(prob, node.subproblem)
    varmap = merge(vin, vout, dmap)
    jl_rows = collect_rows(node.subproblem, varmap)
    jl_obj = collect_obj(node.subproblem, varmap)

    @printf("%-28s %10d %10d\n", "real columns",
            dump["n_cols"], JuMP.num_variables(node.subproblem))
    @printf("%-28s %10d %10d\n", "real structural rows",
            dump["n_rows"], length(jl_rows))

    rt = row_table(dump["rows"], jl_rows)
    println()
    println("3) 真实阶段逐行对比（146 行）")
    @printf("common rows          = %d\n", rt.n_common)
    @printf("missing in Julia     = %s\n", isempty(rt.missing_jl) ? "none" : join(rt.missing_jl, ", "))
    @printf("extra in Julia       = %s\n", isempty(rt.extra_jl) ? "none" : join(rt.extra_jl, ", "))
    @printf("lb max|Δ|            = %.3e\n", rt.max_lb)
    @printf("ub max|Δ|            = %.3e\n", rt.max_ub)
    @printf("coeff key mismatches = %d %s\n", length(rt.key_mismatch),
            isempty(rt.key_mismatch) ? "" : join(rt.key_mismatch, ", "))
    @printf("coeff max|Δ|         = %.3e\n", rt.max_c)

    py_obj = Dict{String,Float64}(String(k) => float(v) for (k, v) in dump["obj_coeffs"])
    obj_keys_ok = Set(keys(py_obj)) == Set(keys(jl_obj))
    max_obj = 0.0
    for k in intersect(Set(keys(py_obj)), Set(keys(jl_obj)))
        max_obj = max(max_obj, abs(py_obj[k] - jl_obj[k]))
    end
    println()
    println("4) 目标系数（共享变量；theta/bellman term 除外）")
    @printf("key sets identical   = %s\n", obj_keys_ok ? "yes" : "NO")
    @printf("coeff max|Δ|         = %.3e\n", max_obj)

    py_bounds_all = Dict{String,NTuple{2,Float64}}(
        String(k) => (float(v[1]), float(v[2])) for (k, v) in dump["var_bounds"])
    jl_bounds = collect_bounds(node.subproblem, varmap)
    shared_names = Set(values(varmap))
    bounds_keys_ok = shared_names == Set(keys(jl_bounds)) &&
                     issubset(shared_names, Set(keys(py_bounds_all)))
    max_lb = 0.0; max_ub = 0.0
    for k in shared_names
        p, j = py_bounds_all[k], jl_bounds[k]
        max_lb = max(max_lb, abs(p[1] - j[1]))
        max_ub = max(max_ub, abs(p[2] - j[2]))
    end
    println()
    println("4b) 变量界（共享变量；theta/bellman term 除外）")
    @printf("shared keys identical = %s\n", bounds_keys_ok ? "yes" : "NO")
    @printf("lb max|Δ|             = %.3e\n", max_lb)
    @printf("ub max|Δ|             = %.3e\n", max_ub)

    # ---- 变量取值与阶段成本（默认求解器 = COPT） ----
    xin, xout = collect_x(prob, node)
    sub = node.subproblem
    taup, taum, send, y, Nvals = collect_decisions(prob, sub)
    stage_cost = stage_cost_of(prob, xin, xout, taup, taum, y)
    grad_copt = collect_grad(prob, node)

    println()
    println("5) 同一状态/场景下的变量取值（软比较；q.out 等可能退化）")
    @printf("%-12s %14s\n", "group", "max|Δ|")
    @printf("%-12s %14.3e\n", "x_in", maxdiff(dump["x_in"], xin))
    @printf("%-12s %14.3e\n", "x_out", maxdiff(dump["x_out"], xout))
    off_o2 = prob.N_R
    off_n2 = off_o2 + (prob.L - 1) * prob.N_R
    off_q2 = off_n2 + prob.N_R * prob.G
    for (blk, rng) in (("I", 1:prob.N_R),
                       ("o", (off_o2 + 1):off_n2),
                       ("n", (off_n2 + 1):off_q2),
                       ("q", (off_q2 + 1):(off_q2 + prob.N_R)))
        @printf("  x_out[%-6s] %14.3e\n", blk, maxdiff(dump["x_out"][rng], xout[rng]))
    end
    @printf("%-12s %14.3e\n", "taup", maxdiff(dump["taup"], taup))
    @printf("%-12s %14.3e\n", "taum", maxdiff(dump["taum"], taum))
    @printf("%-12s %14.3e\n", "send", maxdiff(dump["send"], send))
    @printf("%-12s %14.3e\n", "y", maxdiff(dump["y"], y))
    @printf("%-12s %14.3e\n", "N", maxdiff(dump["N"], Nvals))
    @printf("stage_cost: python %.10f  julia %.10f  |Δ| %.3e\n",
            dump["stage_cost"], stage_cost, abs(dump["stage_cost"] - stage_cost))

    # ---- 同一节点换 HiGHS 重解：目标值与对偶（状态梯度）对比 ----
    JuMP.set_optimizer(sub, HiGHS.Optimizer)
    JuMP.set_silent(sub)
    JuMP.optimize!(sub)
    JuMP.primal_status(sub) == JuMP.MOI.FEASIBLE_POINT ||
        error("HiGHS 重解未得到可行解：$(JuMP.termination_status(sub))")
    xin_h, xout_h = collect_x(prob, node)
    taup_h, taum_h, send_h, y_h, Nvals_h = collect_decisions(prob, sub)
    stage_cost_h = stage_cost_of(prob, xin_h, xout_h, taup_h, taum_h, y_h)
    grad_h = collect_grad(prob, node)
    py_grad = Float64.(dump["x_grad"])

    println()
    println("6) 求解器/对偶差异（固定 x_in 的既约成本 = ∂Q/∂x，Benders 割斜率）")
    @printf("stage_cost COPT=%.10f  HiGHS=%.10f  |Δ|=%.3e\n",
            stage_cost, stage_cost_h, abs(stage_cost - stage_cost_h))
    @printf("max|x_grad python(COPT)|  = %.3e\n", maximum(abs.(py_grad)))
    @printf("py(COPT) vs julia(COPT):   max|Δ|=%.3e  mean|Δ|=%.3e\n",
            maxdiff(py_grad, grad_copt), mean(abs.(py_grad .- grad_copt)))
    @printf("py(COPT) vs julia(HiGHS):  max|Δ|=%.3e  mean|Δ|=%.3e\n",
            maxdiff(py_grad, grad_h), mean(abs.(py_grad .- grad_h)))
    @printf("julia(COPT) vs julia(HiGHS): max|Δ|=%.3e  mean|Δ|=%.3e\n",
            maxdiff(grad_copt, grad_h), mean(abs.(grad_copt .- grad_h)))
    dvec = abs.(py_grad .- grad_h)
    iworst = argmax(dvec)
    n_o = (prob.L - 1) * prob.N_R
    n_n = prob.N_R * prob.G
    blk = iworst <= prob.N_R ? "I" :
          iworst <= prob.N_R + n_o ? "o" :
          iworst <= prob.N_R + n_o + n_n ? "n" : "q"
    @printf("worst py(COPT)-HiGHS: idx=%d block=%s  py=%.6f  highs=%.6f\n",
            iworst, blk, py_grad[iworst], grad_h[iworst])

    # ---- 硬门槛 ----
    ok = virt_names_ok &&
         dump["virtual_n_cols"] == JuMP.num_variables(node1.subproblem) &&
         dump["virtual_n_rows"] == length(rows1) &&
         dump["n_cols"] == JuMP.num_variables(node.subproblem) &&
         dump["n_rows"] == length(jl_rows) &&
         isempty(rt.missing_jl) && isempty(rt.extra_jl) &&
         isempty(rt.key_mismatch) && rt.max_lb <= 1e-9 && rt.max_ub <= 1e-9 &&
         rt.max_c <= 1e-9 && obj_keys_ok && max_obj <= 1e-9 &&
         bounds_keys_ok && max_lb <= 1e-9 && max_ub <= 1e-9 &&
         abs(dump["stage_cost"] - stage_cost) <= 1e-6 &&
         abs(stage_cost - stage_cost_h) <= 1e-6
    println()
    println(ok ? "等价性检查通过（结构 + 行系数 + 目标 + 阶段成本）。"
               : "等价性检查失败，见上表。")
    return ok
end

main() || exit(1)
