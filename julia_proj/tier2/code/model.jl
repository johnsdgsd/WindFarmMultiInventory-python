# model.jl — Tier2 full-integer grade model, exactly the Python Tier2 subproblem
# (code/subproblem.py with integer_scope="tier2_full"), expressed with SDDP.jl.
#
# Graph: SDDP.LinearPolicyGraph(stages = T + 1).
#   node 1      virtual first stage: pin I/o/n, choose q_1, zero stage cost.
#   node 2..T+1 real periods 1..T: all logistics + transition + stage cost.
#
# Tier2 vs V2 (the Python continuous relaxation):
#   - every state component (I,o,n,q) and every control (tau,y,S_end,N) is Int;
#   - N is built only on valid arcs j >= i and replaces the per-arc Ndef rows;
#   - each source grade has Nsum: sum_j N[r,i,j] = n[r,i].in;
#   - each cumulative tail has a nearest-integer band:
#       -0.5 <= sum_{k>=j} N[r,i,k] - C[r,i,j] * n[r,i].in <= 0.5,
#     where C = tail_cdf(Theta). Theta is only a coefficient on n.in.
#   - the logistics rows, stage cost and q timing are identical to V2.
#
# There is no min/max/floor in any state transition. State bounds are explicit
# on .out (JuMP variable bounds); because SDDP.jl 1.13 applies the declared
# bounds/integrality only to .out, explicit .in constraints are added as well
# (redundant with .out bounds at integer states, but needed when a duality
# handler such as LagrangianDuality releases .in).
#
# Python declares x_in as integer variables with bounds [0, ub]. The SDDP.jl
# representation is: .out is declared Int with the same [0, ub] bounds, .in is
# pinned to the previous integer .out in every solve, and the 80 statein_*
# rows reproduce the x_in bounds on .in. At a fixed integer state these rows
# are satisfied and do not restrict any other variable; the one-stage feasible
# set is identical to Python's (verified with incoming_bounds=true vs false in
# test/validate_equivalence.jl).

using SDDP, JuMP, HiGHS, COPT

"""Build the Tier2 policy graph. `optimizer` defaults to `COPT.Optimizer`
(the Python/reference solver); pass `optimizer = HiGHS.Optimizer` for the
open-source cross-check. Use `SDDP.ContinuousConicDuality()` (the default in
`SDDP.train`) for the Python-Tier2 backward LP relaxation, or
`SDDP.LagrangianDuality()` for SDDiP-style cuts."""
function build_sddp_model(prob::ProblemData;
                          optimizer = COPT.Optimizer,
                          lower_bound::Float64 = 0.0,
                          incoming_bounds::Bool = true)
    T, G, N_R, L = prob.T, prob.G, prob.N_R, prob.L
    L >= 2 || error("当前实现要求 L ≥ 2（在途管 L−1 ≥ 1）")
    prob.n0_mode == "integer_largest_remainder" ||
        error("Tier2 模型要求 n0_mode=integer_largest_remainder")
    for r in 1:N_R
        sum(prob.n0[r, :]) == prob.N[r] ||
            error("仓库 $(r) 的 n0 和 $(sum(prob.n0[r, :])) ≠ N=$(prob.N[r])")
        any(prob.n0[r, :] .< 0) && error("仓库 $(r) 的 n0 出现负值")
    end

    model = SDDP.LinearPolicyGraph(
        stages = T + 1,
        sense = :Min,
        lower_bound = lower_bound,
        optimizer = optimizer,
    ) do sp, t
        # ---- state vector order: I | o | n | q (all integer) ----
        @variable(sp, 0 <= I[r = 1:N_R] <= prob.Imax[r],
                  SDDP.State, Int, initial_value = prob.I0[r])
        @variable(sp, 0 <= o[l = 1:L-1, r = 1:N_R] <= prob.Qmax[r],
                  SDDP.State, Int, initial_value = prob.o0[l, r])
        @variable(sp, 0 <= n[r = 1:N_R, i = 1:G] <= prob.N[r],
                  SDDP.State, Int, initial_value = prob.n0[r, i])
        @variable(sp, 0 <= q[r = 1:N_R] <= prob.Qmax[r],
                  SDDP.State, Int, initial_value = 0.0)

        # SDDP.jl only puts the declared bounds/integrality on .out. Add the
        # incoming-state bounds as constraints (Python puts them on x_in).
        if incoming_bounds
            for r in 1:N_R
                @constraint(sp, I[r].in >= 0.0, base_name = "statein_I_lo_$(r)")
                @constraint(sp, I[r].in <= prob.Imax[r],
                            base_name = "statein_I_hi_$(r)")
                @constraint(sp, q[r].in >= 0.0, base_name = "statein_q_lo_$(r)")
                @constraint(sp, q[r].in <= prob.Qmax[r],
                            base_name = "statein_q_hi_$(r)")
            end
            for l in 1:(L-1), r in 1:N_R
                @constraint(sp, o[l, r].in >= 0.0,
                            base_name = "statein_o_lo_$(l)_$(r)")
                @constraint(sp, o[l, r].in <= prob.Qmax[r],
                            base_name = "statein_o_hi_$(l)_$(r)")
            end
            for r in 1:N_R, i in 1:G
                @constraint(sp, n[r, i].in >= 0.0,
                            base_name = "statein_n_lo_$(r)_$(i)")
                @constraint(sp, n[r, i].in <= prob.N[r],
                            base_name = "statein_n_hi_$(r)_$(i)")
            end
        end

        if t == 1
            # ---- virtual first stage: pin the incoming state, decide q_1 ----
            for r in 1:N_R
                @constraint(sp, I[r].out == I[r].in, base_name = "virt_I_$(r)")
            end
            for l in 1:(L-1), r in 1:N_R
                @constraint(sp, o[l, r].out == o[l, r].in,
                            base_name = "virt_o_$(l)_$(r)")
            end
            for r in 1:N_R, i in 1:G
                @constraint(sp, n[r, i].out == n[r, i].in,
                            base_name = "virt_n_$(r)_$(i)")
            end
            # (2) end-of-period capacity, same out-side position as Python.
            for r in 1:N_R
                @constraint(sp,
                    I[r].out + sum(o[l, r].out for l in 1:(L-1)) + q[r].out
                    <= prob.Imax[r],
                    base_name = "cap_$(r)")
            end
            @stageobjective(sp, 0.0)
            return
        end

        # ---- logistics controls (all integer) ----
        @variable(sp, 0 <= taup[r = 1:N_R] <=
                      min(sum(prob.Imax .+ prob.Qmax), prob.Imax[r]),
                  Int, base_name = "taup")
        @variable(sp, 0 <= taum[r = 1:N_R] <= prob.Imax[r] + prob.Qmax[r],
                  Int, base_name = "taum")
        @variable(sp, 0 <= y[r = 1:N_R, m = 1:N_R] <=
                      min(prob.N[r], prob.Imax[m]),
                  Int, base_name = "y")
        @variable(sp, 0 <= S_end[r = 1:N_R] <= prob.Imax[r] + prob.Qmax[r],
                  Int, base_name = "S_end")
        tau = [taup[r] - taum[r] for r in 1:N_R]

        # ---- transition flows N[r,i,j], only valid arcs j >= i ----
        Nflow = Dict{Tuple{Int,Int,Int},JuMP.VariableRef}()
        for r in 1:N_R, i in 1:G, j in i:(G+1)
            Nflow[(r, i, j)] = @variable(sp,
                lower_bound = 0.0, upper_bound = prob.N[r], integer = true,
                base_name = "N_$(r)_$(i)_$(j)")
        end

        # ---- (Tier2) Nsum: one row per source grade ----
        for r in 1:N_R, i in 1:G
            @constraint(sp,
                sum(Nflow[(r, i, j)] for j in i:(G+1)) == n[r, i].in,
                base_name = "Nsum_$(r)_$(i)")
        end

        # ---- (Tier2) cumulative-tail nearest-integer bands ----
        # Placeholder coefficient -1 on n.in; SDDP.parameterize replaces it by
        # -C[r,i,j] = -sum_{k>=j} Theta[r,i,k].
        tail = Dict{Tuple{Int,Int,Int},JuMP.ConstraintRef}()
        for r in 1:N_R, i in 1:G, j in (i+1):(G+1)
            tail[(r, i, j)] = @constraint(sp,
                -0.5 <= sum(Nflow[(r, i, k)] for k in j:(G+1)) - n[r, i].in
                     <= 0.5,
                base_name = "Tail_$(r)_$(i)_$(j)")
        end

        # ---- next-grade column sums ----
        for r in 1:N_R
            @constraint(sp,
                n[r, 1].out == sum(y[r, m] for m in 1:N_R) + Nflow[(r, 1, 1)],
                base_name = "grade0_$(r)")
            for j in 2:G
                @constraint(sp,
                    n[r, j].out == sum(Nflow[(r, i, j)] for i in 1:j),
                    base_name = "grade_$(r)_$(j)")
            end
        end

        # ---- demand / fleet ----
        for r in 1:N_R
            @constraint(sp,
                sum(y[r, m] for m in 1:N_R) +
                sum(Nflow[(r, i, j)] for i in 1:G for j in i:G)
                <= prob.N[r] + prob.fleet_tol,
                base_name = "dem_$(r)")
            @constraint(sp,
                sum(n[r, i].in for i in 1:G) <= prob.N[r] + prob.fleet_tol,
                base_name = "fleet_$(r)")
        end

        # ---- passive transfer, active transfer, inventory, pipeline ----
        for m in 1:N_R
            @constraint(sp, sum(y[r, m] for r in 1:N_R) <= I[m].in,
                        base_name = "pas_$(m)")
        end
        for r in 1:N_R
            @constraint(sp,
                S_end[r] == I[r].in - sum(y[s, r] for s in 1:N_R) + o[1, r].in,
                base_name = "send_def_$(r)")
        end
        @constraint(sp, sum(tau[r] for r in 1:N_R) == 0.0,
                    base_name = "tau_bal")
        for r in 1:N_R
            @constraint(sp, S_end[r] + tau[r] >= 0.0,
                        base_name = "tau_lo_$(r)")
            @constraint(sp, S_end[r] + tau[r] <= prob.Imax[r],
                        base_name = "tau_hi_$(r)")
            @constraint(sp, tau[r] <=
                            sum(S_end[m] for m in 1:N_R if m != r),
                        base_name = "tau_src_$(r)")
            @constraint(sp, I[r].out == S_end[r] + tau[r],
                        base_name = "inv_$(r)")
        end
        for l in 1:(L-2), r in 1:N_R
            @constraint(sp, o[l, r].out == o[l+1, r].in,
                        base_name = "pipe_$(l)_$(r)")
        end
        for r in 1:N_R
            @constraint(sp, o[L-1, r].out == q[r].in,
                        base_name = "pipe_last_$(r)")
            @constraint(sp,
                I[r].out + sum(o[l, r].out for l in 1:(L-1)) + q[r].out
                <= prob.Imax[r],
                base_name = "cap_$(r)")
        end

        # ---- stage cost: purchase + holding + downtime + active + passive ----
        ordering = prob.C * sum(q[r].in for r in 1:N_R)
        holding = prob.h / 2.0 * sum(I[r].in + I[r].out for r in 1:N_R)
        downtime = prob.Cs * sum(prob.N[r] - sum(n[r, i].in for i in 1:G)
                                 for r in 1:N_R)
        active = prob.atc / 2.0 * sum(taup[r] + taum[r] for r in 1:N_R)
        passive = sum(prob.tc_route[r, m] * y[r, m]
                      for r in 1:N_R for m in 1:N_R)
        @stageobjective(sp, ordering + holding + downtime + active + passive)

        # ---- noise: exact exported (z, Theta); equal weights ----
        SDDP.parameterize(sp, prob.pool,
                          fill(1.0 / length(prob.pool), length(prob.pool))) do (z, theta)
            for r in 1:N_R, i in 1:G, j in (i+1):(G+1)
                c = sum(theta[r, i, k] for k in j:(G+1))
                JuMP.set_normalized_coefficient(tail[(r, i, j)],
                                                n[r, i].in, -c)
            end
        end
    end
    return model
end
