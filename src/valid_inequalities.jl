
"""
Compute the upper bounds on `A_i ⋅ x under common linear constraints`
"""
function compute_bounds(bp::BilevelProblem; optimizer, silent=true, iterated=true)
    m = Model(optimizer)
    silent && MOI.set(m, MOI.Silent(), true)
    (_, x, v, λ, σ, s, _) = high_point_relaxation(m, bp)
    optimize!(m)
    st = termination_status(m)
    @assert st == MOI.OPTIMAL
    function bound_vector()
        bounds_val = Float64[]
        for i in 1:bp.ml
            @objective(m, Max, bp.A[i,:] ⋅ x)
            optimize!(m)
            st = termination_status(m)
            if st == MOI.INFEASIBLE || st == MOI.INFEASIBLE_OR_UNBOUNDED
                @warn "Status $st"
                return (st, [NaN])
            elseif st == MOI.DUAL_INFEASIBLE
                push!(bounds_val, Inf)
            else
                st == MOI.OPTIMAL || error("Status $st")
                push!(bounds_val, objective_value(m))
            end
        end
        return (MOI.OPTIMAL, bounds_val)
    end
    (st, Abounds) = bound_vector()
    if st != MOI.OPTIMAL
        return (st, Abounds)
    end
    niter = 1
    while iterated && all(isfinite, Abounds)
        @info niter
        Aprev = copy(Abounds)
        add_bound_inequalities(m, bp, x, v, λ, Aprev)
        (st, Abounds) = bound_vector()
        if st != MOI.OPTIMAL
            return (st, Abounds)
        end
        if all(tup -> tup[1] ≈ tup[2], zip(Abounds, Aprev))
            break
        end
        niter += 1
    end
    return (MOI.OPTIMAL, Abounds)
end

function add_bound_inequalities(m::JuMP.Model, bp::BilevelProblem, x, v, λ, bounds::AbstractVector{<:Real})
    @constraint(m, λ ⋅ bp.b + v ⋅ bp.d ≤ bounds ⋅ λ)
    return nothing
end

function add_bound_inequalities(m::JuMP.Model, bp::BilevelProblem, x, v, λ; optimizer, silent=true, verbose=false)
    (st, Aplus) = compute_bounds(bp; optimizer=optimizer, silent=silent)
    if st != MOI.OPTIMAL
        return false
    end
    if all(isfinite, Aplus)
        add_bound_inequalities(m, bp, x, v, λ, Aplus)
        return true
    end
    verbose && @warn("Not all bounds are finite: # inf: $(count(isfinite, Aplus))")
    return false
end
