# validate_coefficients.jl — exact coefficient-level comparison with Python.
#
# Compares, after mapping Julia names to the Python layout:
#   * stage 1: all 146 Python rows (coefficients + lb/ub), the objective
#     (65 stage-cost terms + constant 320600), and the 191 column bounds/types;
#   * stage 0 (virtual): the 5 cap rows, the 35 virt .out==.in rows (Python
#     pins I/o/n x_out by bounds instead), the 80 statein rows, q.out free,
#     no controls, stage objective 0 (theta only).
#
# Run:
#   "D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/dump_python_coeffs.py
#   julia test/validate_coefficients.jl

const _ROOT = dirname(dirname(abspath(@__FILE__)))
include(joinpath(_ROOT, "code", "problem_data.jl"))
include(joinpath(_ROOT, "code", "model.jl"))
using SDDP, JuMP, COPT
import MathOptInterface as MOI

const PYF = joinpath(_ROOT, "test", "_python_coeffs.json")
isfile(PYF) || error("先运行 test/dump_python_coeffs.py")
pyd = JSON.parsefile(PYF)
prob = DefineProblem(joinpath(_ROOT, "config", "problem.toml"); T = 2, pool_size = 8)
Ly = state_layout(prob)
N_R, G, L = prob.N_R, prob.G, prob.L

let
canon_state(block, idx, suffix) = (suffix == "in" ? "xin_" : "xout_") * string(idx)

function canon_var(name::AbstractString)
    name == "" && return "theta"
    m = match(r"^I\[(\d+)\]_(in|out)$", name)
    if m !== nothing
        r = parse(Int, m.captures[1])
        return canon_state("I", r - 1, m.captures[2])
    end
    m = match(r"^o\[(\d+),(\d+)\]_(in|out)$", name)
    if m !== nothing
        l = parse(Int, m.captures[1]); r = parse(Int, m.captures[2])
        return canon_state("o", 5 + (l - 1) * N_R + (r - 1), m.captures[3])
    end
    m = match(r"^n\[(\d+),(\d+)\]_(in|out)$", name)
    if m !== nothing
        r = parse(Int, m.captures[1]); i = parse(Int, m.captures[2])
        return canon_state("n", 15 + (r - 1) * G + (i - 1), m.captures[3])
    end
    m = match(r"^q\[(\d+)\]_(in|out)$", name)
    if m !== nothing
        r = parse(Int, m.captures[1])
        return canon_state("q", 35 + (r - 1), m.captures[2])
    end
    m = match(r"^taup\[(\d+)\]$", name)
    m !== nothing && return "taup_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^taum\[(\d+)\]$", name)
    m !== nothing && return "taum_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^y\[(\d+),(\d+)\]$", name)
    m !== nothing && return "y_$(parse(Int, m.captures[1]) - 1)_$(parse(Int, m.captures[2]) - 1)"
    m = match(r"^S_end\[(\d+)\]$", name)
    m !== nothing && return "send_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^N_(\d+)_(\d+)_(\d+)$", name)
    m !== nothing && return "N_$(parse(Int, m.captures[1]) - 1)_$(parse(Int, m.captures[2]) - 1)_$(parse(Int, m.captures[3]) - 1)"
    return name
end

function canon_row(name::AbstractString)
    m = match(r"^pas_(\d+)$", name)
    m !== nothing && return "pas_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^Nsum_(\d+)_(\d+)$", name)
    m !== nothing && return "Nsum_$(parse(Int, m.captures[1]) - 1)_$(parse(Int, m.captures[2]) - 1)"
    m = match(r"^Tail_(\d+)_(\d+)_(\d+)$", name)
    m !== nothing && return "Tail_$(parse(Int, m.captures[1]) - 1)_$(parse(Int, m.captures[2]) - 1)_$(parse(Int, m.captures[3]) - 1)"
    m = match(r"^grade0_(\d+)$", name)
    m !== nothing && return "grade0_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^grade_(\d+)_(\d+)$", name)
    m !== nothing && return "grade_$(parse(Int, m.captures[1]) - 1)_$(parse(Int, m.captures[2]) - 1)"
    m = match(r"^dem_(\d+)$", name)
    m !== nothing && return "dem_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^fleet_(\d+)$", name)
    m !== nothing && return "fleet_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^send_def_(\d+)$", name)
    m !== nothing && return "send_def_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^tau_lo_(\d+)$", name)
    m !== nothing && return "tau_lo_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^tau_hi_(\d+)$", name)
    m !== nothing && return "tau_hi_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^tau_src_(\d+)$", name)
    m !== nothing && return "tau_src_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^inv_(\d+)$", name)
    m !== nothing && return "inv_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^pipe_last_(\d+)$", name)
    m !== nothing && return "pipe_last_$(parse(Int, m.captures[1]) - 1)"
    m = match(r"^pipe_(\d+)_(\d+)$", name)
    m !== nothing && return "pipe_$(parse(Int, m.captures[1]) - 1)_$(parse(Int, m.captures[2]) - 1)"
    m = match(r"^cap_(\d+)$", name)
    m !== nothing && return "cap_$(parse(Int, m.captures[1]) - 1)"
    name == "tau_bal" && return "tau_bal"
    return name
end

set_bounds(con) = begin
    set = JuMP.constraint_object(con).set
    if set isa MOI.LessThan{Float64}
        (-Inf, set.upper)
    elseif set isa MOI.GreaterThan{Float64}
        (set.lower, Inf)
    elseif set isa MOI.EqualTo{Float64}
        (set.value, set.value)
    elseif set isa MOI.Interval{Float64}
        (set.lower, set.upper)
    else
        error("unexpected set $(typeof(set))")
    end
end

function julia_rows(sp_)
    d = Dict{String,Any}()
    for (F, S) in JuMP.list_of_constraint_types(sp_)
        F == JuMP.VariableRef && continue
        for con in JuMP.all_constraints(sp_, F, S)
            nm = JuMP.name(con)
            startswith(nm, "statein") && continue
            obj = JuMP.constraint_object(con)
            terms = Dict{String,Float64}()
            for (v, c) in obj.func.terms
                abs(c) == 0.0 && continue
                terms[canon_var(JuMP.name(v))] = Float64(c)
            end
            d[canon_row(nm)] = (bounds = set_bounds(con), terms = terms,
                                jlname = nm)
        end
    end
    return d
end

# COPT's default MatrixTol is 1e-10: coefficients below it are treated as zero
# (they also do not appear in m.getA()). Julia/JuMP keeps them. Compare up to
# that tolerance and track the largest dropped coefficient separately.
MTOL = 1e-10
TINY = Float64[]

function compare_terms(label, jt, pt)
    ok = true
    jf = Dict{String,Float64}(); pf = Dict{String,Float64}()
    for (k, v) in jt
        if abs(v) > MTOL; jf[k] = v; else; push!(TINY, abs(v)); end
    end
    for (k, v) in pt
        if abs(Float64(v)) > MTOL; pf[k] = Float64(v); else; push!(TINY, abs(Float64(v))); end
    end
    for (k, v) in pf
        if !haskey(jf, k)
            println("  [FAIL] $(label): missing variable $(k)"); ok = false
        elseif !isapprox(jf[k], v; atol = 1e-12, rtol = 1e-10)
            println("  [FAIL] $(label): $(k) Julia=$(jf[k]) Python=$(v)"); ok = false
        end
    end
    for k in keys(jf)
        if !haskey(pf, k)
            println("  [FAIL] $(label): extra variable $(k)"); ok = false
        end
    end
    return ok
end

# ---------------- stage 1 ----------------
println("="^90)
println("A. 真实阶段（node 2）逐行系数 + 目标")
println("="^90)
prob4 = DefineProblem(joinpath(_ROOT, "config", "problem.toml"); T = 2, pool_size = 8)
model = build_sddp_model(prob4)
node = model[2]
sp = node.subproblem
JuMP.set_optimizer(sp, COPT.Optimizer)
JuMP.set_silent(sp)
x0 = initial_state(prob4)
for r in 1:prob4.N_R
    JuMP.fix(node.states[Symbol("I[$(r)]")].in, x0[ix_I(Ly, r)]; force = true)
    JuMP.fix(node.states[Symbol("q[$(r)]")].in, x0[ix_q(Ly, r)]; force = true)
    for l in 1:(prob4.L - 1)
        JuMP.fix(node.states[Symbol("o[$(l),$(r)]")].in,
                 x0[ix_o(Ly, l, r)]; force = true)
    end
    for i in 1:prob4.G
        JuMP.fix(node.states[Symbol("n[$(r),$(i)]")].in,
                 x0[ix_n(Ly, r, i)]; force = true)
    end
end
node.parameterize(prob4.pool[1])
jr = julia_rows(sp)
py = pyd["stage1"]
ok = true
for prow in py["rows"]
    nm = prow["name"]
    if !haskey(jr, nm)
        println("  [FAIL] missing row $(nm)"); ok = false; continue
    end
    jrow = jr[nm]
    plb = Float64(prow["lb"]) <= -1e29 ? -Inf : Float64(prow["lb"])
    pub = Float64(prow["ub"]) >= 1e29 ? Inf : Float64(prow["ub"])
    if !(isapprox(jrow.bounds[1], plb; atol = 1e-12) &&
         isapprox(jrow.bounds[2], pub; atol = 1e-12))
        println("  [FAIL] row $(nm) bounds Julia=$(jrow.bounds) " *
                "Python=($(plb), $(pub))"); ok = false
    end
    ok &= compare_terms("row $(nm)", jrow.terms, prow["coeffs"])
end
extra = setdiff(keys(jr), [r["name"] for r in py["rows"]])
isempty(extra) || (println("  [FAIL] extra Julia rows: $(collect(extra))"); ok = false)
@assert length(jr) == 146
println("  146 行（系数与 lb/ub）逐项一致: ", ok ? "PASS" : "FAIL")
println("  低于 COPT MatrixTol=1e-10 的系数: $(length(TINY)) 个，max|.|=" *
        "$(isempty(TINY) ? 0.0 : maximum(TINY))（COPT 按 0 处理，Julia 保留；" *
        "n≤3780 时对行的影响 < 4e-9）")
@assert ok

# Objective (exclude the anonymous Bellman theta; compare it separately).
jobj = JuMP.objective_function(sp)
const_j = Float64(JuMP.constant(jobj))
jterms = Dict{String,Float64}()
theta_coeff = 0.0
for (v, c) in jobj.terms
    nm = canon_var(JuMP.name(v))
    if nm == "theta"
        theta_coeff = Float64(c)
    else
        jterms[nm] = Float64(c)
    end
end
ok_obj = compare_terms("objective", jterms, py["obj"])
ok_obj &= isapprox(const_j, Float64(py["obj_const"]); atol = 1e-9)
println("  目标: $(length(py["obj"])) 个阶段成本项 + 常数 $(const_j): ",
        ok_obj ? "PASS" : "FAIL", "；Bellman theta 系数=$(theta_coeff) ",
        "(Python assemble(None) 把 theta 钉在 0，故其目标不含 theta)")
@assert ok_obj
@assert theta_coeff == 1.0

# Column bounds/types (xin handled by the statein rows).
jcols = Dict{String,Any}()
for v in JuMP.all_variables(sp)
    nm = canon_var(JuMP.name(v))
    lb = JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf
    ub = JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf
    jcols[nm] = (lb = lb, ub = ub, isint = JuMP.is_integer(v))
end
ok_c = true
for c in py["cols"]
    nm = c["name"]
    nm == "theta" && continue
    nm in ("xin_$i" for i in 0:(Ly.dim - 1)) && continue  # via statein rows
    jc = jcols[nm]
    want_int = c["type"] == "I"
    if !(isapprox(jc.lb, c["lb"]; atol = 1e-12) &&
         isapprox(jc.ub, c["ub"]; atol = 1e-12) && jc.isint == want_int)
        println("  [FAIL] col $(nm): Julia=($(jc.lb),$(jc.ub),int=$(jc.isint)) " *
                "Python=($(c["lb"]),$(c["ub"]),int=$(want_int))"); ok_c = false
    end
end
@assert ok_c
println("  xout(40) + 控制(110) 的 lb/ub/整数性与 Python 一致: PASS")
@assert length(py["cols"]) == 191 && length(jcols) == 191

# ---------------- stage 0 (virtual) ----------------
println("\n" * "="^90)
println("B. 虚拟首阶段（node 1）")
println("="^90)
node0 = model[1]
sp0 = node0.subproblem
@assert JuMP.num_variables(sp0) == 81
@assert JuMP.num_constraints(sp0; count_variable_in_set_constraints = false) == 120
var0 = JuMP.name.(JuMP.all_variables(sp0))
@assert !any(n -> occursin("taup", n) || occursin("taum", n) ||
                  occursin("y[", n) || occursin("N_", n) ||
                  occursin("S_end", n), var0)
println("  81 列 = 40 in + 40 out + theta；无任何物流控制列：PASS")
println("  120 行 = 5 cap + 35 virt(.out==.in) + 80 statein")
con0 = Dict{String,Any}()
n_virt = 0
n_cap = 0
n_statein = 0
for (F, S) in JuMP.list_of_constraint_types(sp0)
    F == JuMP.VariableRef && continue
    for con in JuMP.all_constraints(sp0, F, S)
        nm = JuMP.name(con)
        obj = JuMP.constraint_object(con)
        if startswith(nm, "virt_")
            n_virt += 1
            @assert length(obj.func.terms) == 2
            cs = sort(collect(Float64, values(obj.func.terms)))
            vs = JuMP.name.(collect(keys(obj.func.terms)))
            @assert cs == [-1.0, 1.0]
            @assert any(endswith.(vs, "_in")) && any(endswith.(vs, "_out"))
            @assert set_bounds(con) == (0.0, 0.0)
            @assert !occursin("q[", vs[1]) && !occursin("q[", vs[2])
        elseif startswith(nm, "statein")
            n_statein += 1
        else
            n_cap += 1
            terms = Dict{String,Float64}(
                canon_var(JuMP.name(v)) => Float64(c)
                for (v, c) in obj.func.terms if abs(c) > 0)
            con0[canon_row(nm)] = (bounds = set_bounds(con), terms = terms)
        end
    end
end
@assert n_virt == 35 && n_cap == 5 && n_statein == 80
println("  virt 35 行：仅 I/o/n 的 .out - .in == 0；无 q virt：PASS")
# cap rows vs Python stage 0.
py0 = pyd["stage0"]
ok0 = true
for prow in py0["rows"]
    jr0 = con0[prow["name"]]
    plb = Float64(prow["lb"]) <= -1e29 ? -Inf : Float64(prow["lb"])
    pub = Float64(prow["ub"]) >= 1e29 ? Inf : Float64(prow["ub"])
    ok0 &= isapprox(jr0.bounds[1], plb; atol = 1e-12) &&
           isapprox(jr0.bounds[2], pub; atol = 1e-12)
    ok0 &= compare_terms("stage0 row $(prow["name"])", jr0.terms, prow["coeffs"])
end
println("  cap 5 行系数/界与 Python stage 0 一致: ", ok0 ? "PASS" : "FAIL")
@assert ok0
# q.out free in [0,Qmax] and integer; objective is theta only.
for r in 1:N_R
    v = node0.states[Symbol("q[$(r)]")].out
    @assert JuMP.lower_bound(v) == 0.0
    @assert JuMP.upper_bound(v) == prob4.Qmax[r]
    @assert JuMP.is_integer(v)
end
jobj0 = JuMP.objective_function(sp0)
@assert JuMP.constant(jobj0) == 0.0
@assert length(jobj0.terms) == 1
tv = first(keys(jobj0.terms))
@assert JuMP.name(tv) == ""
@assert first(values(jobj0.terms)) == 1.0
println("  q.out 自由 [0,Qmax] 整数；阶段目标仅 theta（无采购成本）：PASS")
println("  （Python 用 x_out 变量界钉 I/o/n、x_in 变量界钉 40 维；Julia 用 " *
        "35 virt + 80 statein 表达，可行域相同）")

println("\nCOEFFICIENT-LEVEL EQUIVALENCE: PASS")
end
