# model.jl — build_sddp_model(prob)：T+1 阶段 LinearPolicyGraph（V2 纯 LP）。
#
# 阶段结构（与 Python 引擎一致）：
#   节点 1        虚拟首阶段：I/o/n 用 out==in 钉住初值，只决定首期订货 q_1；
#                 阶段成本 0。q_1 的采购成本在下个节点（真实期 1）按 q.in 计。
#   节点 2..T+1   真实期 1..T：需求实现、被动/主动调拨、库存/在途/等级递推。
#
# 状态布局（与 Python state_layout.py 相同）：I(N_R) | o((L-1)·N_R) |
# n(N_R·G) | q(N_R)，全部连续。状态变量名由 SDDP 生成为 I[r]、o[l,r]、
# n[r,i]、q[r]；决策/约束名刻意沿用 Python 的 0-based 名字
# （taup_r、y_r_m、N_r_i_j、pas_m、Ndef_r_i_j、grade0_r、grade_r_j、…），
# 便于逐行等价性检查。
#
# ★ 噪声处理：Θ 只作 n[r,i].in 的约束系数。
#   基础行 Ndef: N - n_in == 0；SDDP.parameterize 内用
#   set_normalized_coefficient(Ndef, n_in, -Θ) 改成 N - Θ·n_in == 0。
#   禁止把 N 算成“由状态得到的常数”塞进右端：对偶看不见 ∂N/∂n，Benders 割
#   会系统性越界（Python docs/割越界根因分析.md）。也禁止 Θ 乘变量（二次）。
#   所有 Ndef 系数（含结构零 0.0）都必须在 parameterize 内设置，否则会残留
#   上一个场景的系数。
#
# ★ 全部变量连续：没有 Int/Bin，没有 JuMP.fix(F, round(Θn))。这是 V2
#   （Python integer_enabled=false / scope=none）与参考 Julia MIP 的唯一差别。
#
# 约束与 Python code/subproblem.py 一一对应：
#   pas_m:              Σ_r y[r,m] ≤ I[m].in
#   Ndef_r_i_j:         N[r,i,j] − Θ[r,i,j]·n[r,i].in = 0
#   grade0_r:           n[r,1].out = Σ_m y[r,m] + N[r,1,1]
#   grade_r_j (j≥2):    n[r,j].out = Σ_{i=1..j} N[r,i,j]
#   dem_r:              Σ_m y + Σ_{i,j<G} N ≤ N_r + fleet_tol
#   fleet_r:            Σ_i n[r,i].in ≤ N_r + fleet_tol
#   send_def_r:         S_end[r] = I[r].in − Σ_s y[s,r] + o[1,r].in
#   tau_bal / tau_lo_r / tau_hi_r / tau_src_r
#   inv_r:              I[r].out = S_end[r] + τ[r]
#   pipe_l_r / pipe_last_r
#   cap_r:              I[r].out + Σ_l o[l,r].out + q[r].out ≤ Imax[r]
#
# 目标（与 Python 相同，q.in 计费）：
#   C·Σq.in + h/2·Σ(I.in+I.out) + Cs·(ΣN_r − Σn.in)
#   + atc/2·Σ(τ⁺+τ⁻) + Σ tc_route[r,m]·y[r,m]

using SDDP, JuMP, HiGHS

# 默认优先 COPT（与 Python V2 同求解器族）；COPT.jl 不可用时回退 HiGHS。
# 这是求解器层选择，不改变任何方程/目标/状态。
const HAS_COPT = try
    @eval import COPT
    true
catch err
    @warn "COPT.jl 加载失败，V2 默认优化器回退到 HiGHS" exception = (err, catch_backtrace())
    false
end

"""默认优化器：COPT（可用时）或 HiGHS（回退）。"""
default_optimizer() = HAS_COPT ? COPT.Optimizer : HiGHS.Optimizer

function build_sddp_model(prob::ProblemData; optimizer = default_optimizer())
    T, G, N_R, L = prob.T, prob.G, prob.N_R, prob.L
    L >= 2 || error("当前实现要求 L ≥ 2（在途管 L−1 ≥ 1），实际 $L")
    N_r, Imax, Qmax = prob.N, prob.Imax, prob.Qmax
    total_N = sum(N_r)

    # 每个真实期的停机成本 ≥ −Cs·Σ_r fleet_tol；虚拟期成本 0。
    lower_bound = -prob.Cs * prob.fleet_tol * N_R * T

    return SDDP.LinearPolicyGraph(
        stages = T + 1,
        sense = :Min,
        lower_bound = lower_bound,
        optimizer = optimizer,
    ) do sp, t
        # ---- 状态（全部连续；界写在 .out 上，.in 由 SDDP 钉入来值） ----
        @variable(sp, 0 <= I[r = 1:N_R] <= Imax[r],
                  SDDP.State, initial_value = prob.I0[r])
        @variable(sp, 0 <= o[l = 1:(L - 1), r = 1:N_R] <= Qmax[r],
                  SDDP.State, initial_value = 0.0)
        @variable(sp, 0 <= n[r = 1:N_R, i = 1:G] <= N_r[r],
                  SDDP.State, initial_value = prob.n0[r, i])
        @variable(sp, 0 <= q[r = 1:N_R] <= Qmax[r],
                  SDDP.State, initial_value = 0.0)

        # ---------------- 虚拟首阶段 ----------------
        # Python 用 _pin_map 把 I/o/n 的 x_out 直接钉在初值（变量界），只留 q.out
        # 自由；这里用 JuMP.fix 复现同一结构（不额外加 pin 行）。
        if t == 1
            for r in 1:N_R
                JuMP.fix(I[r].out, prob.I0[r]; force = true)
            end
            for l in 1:(L - 1), r in 1:N_R
                JuMP.fix(o[l, r].out, prob.o0[l, r]; force = true)
            end
            for r in 1:N_R, i in 1:G
                JuMP.fix(n[r, i].out, prob.n0[r, i]; force = true)
            end
            c_cap0 = @constraint(sp, [r = 1:N_R],
                I[r].out + sum(o[l, r].out for l in 1:(L - 1)) + q[r].out <= Imax[r])
            for r in 1:N_R
                set_name(c_cap0[r], "cap_$(r - 1)")
            end
            @stageobjective(sp, 0.0)
            return
        end

        # ---------------- 真实期决策 ----------------
        cap_sum = sum(Imax) + sum(Qmax)
        @variable(sp, 0 <= taup[r = 1:N_R] <= min(cap_sum, Imax[r]))
        @variable(sp, 0 <= taum[r = 1:N_R] <= Imax[r] + Qmax[r])
        @variable(sp, 0 <= y[r = 1:N_R, m = 1:N_R] <= min(N_r[r], Imax[m]))
        @variable(sp, 0 <= S_end[r = 1:N_R] <= Imax[r] + Qmax[r])
        @variable(sp, 0 <= N[r = 1:N_R, i = 1:G, j = i:(G + 1)] <= N_r[r])

        for r in 1:N_R
            set_name(taup[r], "taup_$(r - 1)")
            set_name(taum[r], "taum_$(r - 1)")
            set_name(S_end[r], "send_$(r - 1)")
        end
        for r in 1:N_R, m in 1:N_R
            set_name(y[r, m], "y_$(r - 1)_$(m - 1)")
        end
        for r in 1:N_R, i in 1:G, j in i:(G + 1)
            set_name(N[r, i, j], "N_$(r - 1)_$(i - 1)_$(j - 1)")
        end

        @expression(sp, tau[r = 1:N_R], taup[r] - taum[r])

        # ---- 被动调拨（先）：各仓满足总量不超过期初库存 ----
        c_pas = @constraint(sp, [m = 1:N_R],
                            sum(y[r, m] for r in 1:N_R) <= I[m].in)
        for m in 1:N_R
            set_name(c_pas[m], "pas_$(m - 1)")
        end

        # ---- 等级转移流量：N = Θ·n.in（Θ 在 parameterize 中作系数写入） ----
        c_Ndef = @constraint(sp, [r = 1:N_R, i = 1:G, j = i:(G + 1)],
                             N[r, i, j] - n[r, i].in == 0.0)
        for r in 1:N_R, i in 1:G, j in i:(G + 1)
            set_name(c_Ndef[r, i, j],
                     "Ndef_$(r - 1)_$(i - 1)_$(j - 1)")
        end

        # ---- 下一期等级：列和 ----
        c_grade0 = @constraint(sp, [r = 1:N_R],
            n[r, 1].out == sum(y[r, m] for m in 1:N_R) + N[r, 1, 1])
        c_grade = @constraint(sp, [r = 1:N_R, j = 2:G],
            n[r, j].out == sum(N[r, i, j] for i in 1:j))
        for r in 1:N_R
            set_name(c_grade0[r], "grade0_$(r - 1)")
        end
        for r in 1:N_R, j in 2:G
            set_name(c_grade[r, j], "grade_$(r - 1)_$(j - 1)")
        end

        # ---- 需求/待修：替换量 + 期末运行台数 ≤ 机队 ----
        c_dem = @constraint(sp, [r = 1:N_R],
            sum(y[r, m] for m in 1:N_R) +
            sum(N[r, i, j] for i in 1:G, j in i:G) <=
            N_r[r] + prob.fleet_tol)
        # ---- W_t ≥ 0：期初运行台数不超过机队（容差只吸收 LP 残差） ----
        c_fleet = @constraint(sp, [r = 1:N_R],
            sum(n[r, i].in for i in 1:G) <= N_r[r] + prob.fleet_tol)
        for r in 1:N_R
            set_name(c_dem[r], "dem_$(r - 1)")
            set_name(c_fleet[r], "fleet_$(r - 1)")
        end

        # ---- 被动调拨后剩余库存（含本期到货 o[1]） ----
        c_send = @constraint(sp, [r = 1:N_R],
            S_end[r] == I[r].in - sum(y[s, r] for s in 1:N_R) + o[1, r].in)

        # ---- 主动调拨 ----
        c_tau_bal = @constraint(sp, sum(tau[r] for r in 1:N_R) == 0.0,
                                base_name = "tau_bal")
        c_tau_lo = @constraint(sp, [r = 1:N_R], S_end[r] + tau[r] >= 0.0)
        c_tau_hi = @constraint(sp, [r = 1:N_R], S_end[r] + tau[r] <= Imax[r])
        c_tau_src = @constraint(sp, [r = 1:N_R],
            tau[r] <= sum(S_end[m] for m in 1:N_R if m != r))
        for r in 1:N_R
            set_name(c_send[r], "send_def_$(r - 1)")
            set_name(c_tau_lo[r], "tau_lo_$(r - 1)")
            set_name(c_tau_hi[r], "tau_hi_$(r - 1)")
            set_name(c_tau_src[r], "tau_src_$(r - 1)")
        end

        # ---- 库存动态 ----
        c_inv = @constraint(sp, [r = 1:N_R], I[r].out == S_end[r] + tau[r])
        for r in 1:N_R
            set_name(c_inv[r], "inv_$(r - 1)")
        end

        # ---- 在途滚动 ----
        c_pipe = @constraint(sp, [l = 1:(L - 2), r = 1:N_R],
                             o[l, r].out == o[l + 1, r].in)
        c_pipe_last = @constraint(sp, [r = 1:N_R], o[L - 1, r].out == q[r].in)
        for l in 1:(L - 2), r in 1:N_R
            set_name(c_pipe[l, r], "pipe_$(l - 1)_$(r - 1)")
        end
        for r in 1:N_R
            set_name(c_pipe_last[r], "pipe_last_$(r - 1)")
        end

        # ---- 容量（期末侧）：库存 + 在途 + 订货 ≤ Imax ----
        c_cap = @constraint(sp, [r = 1:N_R],
            I[r].out + sum(o[l, r].out for l in 1:(L - 1)) + q[r].out <= Imax[r])
        for r in 1:N_R
            set_name(c_cap[r], "cap_$(r - 1)")
        end

        # ---------------- 噪声：Θ 作 n.in 的系数 ----------------
        weights = fill(1.0 / length(prob.pool), length(prob.pool))
        SDDP.parameterize(sp, prob.pool, weights) do (_, theta)
            for r in 1:N_R, i in 1:G, j in i:(G + 1)
                JuMP.set_normalized_coefficient(
                    c_Ndef[r, i, j], n[r, i].in, -theta[r, i, j])
            end
        end

        # ---------------- 阶段成本 ----------------
        ordering = prob.C * sum(q[r].in for r in 1:N_R)
        holding = prob.h / 2 * sum(I[r].in + I[r].out for r in 1:N_R)
        downtime = prob.Cs * total_N - prob.Cs * sum(n[r, i].in
                                                      for r in 1:N_R, i in 1:G)
        active = prob.atc / 2 * sum(taup[r] + taum[r] for r in 1:N_R)
        passive = sum(prob.tc_route[r, m] * y[r, m]
                      for r in 1:N_R, m in 1:N_R)
        @stageobjective(sp, ordering + holding + downtime + active + passive)
    end
end
