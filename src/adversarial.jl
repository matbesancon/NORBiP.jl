
"""
    near_optimal_robust_violated(bp, x, v; optimizer, [atol])

Returns the set of non near-optimal robust upper-level constraints
"""
function near_optimal_robust_violated(bp::BilevelProblem, x::Vector{Float64}, v::Vector{Float64}, δ; optimizer, atol=10e-4)
    m = Model(optimizer)
    MOI.set(m, MOI.Silent(), true)
    @variable(m, y[1:bp.nl] >= 0)
    @constraint(m, bp.A * x + bp.B * y .≤ bp.b)
    @constraint(m, bp.d ⋅ y ≤ bp.d ⋅ v + δ)
    ks = BitSet()
    for k in 1:bp.mu
        @objective(m, Max, bp.H[k,:] ⋅ y)
        optimize!(m)
        if termination_status(m) == MOI.OPTIMAL || termination_status(m) == MOI.DUAL_INFEASIBLE
            if objective_value(m) > bp.q[k] - bp.G[k,:] ⋅ x - atol
                push!(ks, k)
            end
        end
    end
    return ks
end
