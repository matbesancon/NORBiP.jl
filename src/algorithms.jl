
"""
    solve_near_optimal(m::JuMP.Model, bp::BilevelProblem, ::ExtendedMethod, δ::Real, vertex_list;...)

Solve NORBiP using the extended formulation.
Creates a disjunctive constraint over vertices of each dual polyhedron using indicator constraints.

`vertex_list` is a vector of vectors of length `bp.mu`, with each
individual sub-vector containing a list of tuples `(α, β)` with α a vector of `bp.ml` elements and β a scalar.
"""
function solve_near_optimal(m::JuMP.Model, bp::BilevelProblem, ::ExtendedMethod, δ::Real, vertex_list; verbose=false, resvar=false)
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds) = bilevel_optimality(m, bp, upperlevel=false, resvar=resvar)
    # early termination if relaxed or bilevel optimistic infeasible
    optimize!(m)
    st = JuMP.termination_status(m)
    if st != MOI.OPTIMAL && st != MOI.DUAL_INFEASIBLE
        return (m, st)
    end
    w = Vector{Vector{VariableRef}}(undef, bp.mu)
    for k in 1:bp.mu
        w[k] = @variable(m, [l=1:length(vertex_list[k])], Bin, base_name="w_$(k)_")
        @constraint(m, sum(w[k]) >= 1)
        for l in eachindex(vertex_list[k])
            (α, β) = vertex_list[k][l]
            first_part = if resvar
                dot(α, m[:r])
            else
                dot(α, bp.b - bp.A * x)
            end
            @constraint(m, [w[k][l], first_part + β * dot(bp.d, v) + dot(bp.G[k,:], x)]
                in MOI.IndicatorSet{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(bp.q[k] - β * δ))
            )
        end
    end
    return (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds, w)
end

function bilevel_optimality(m::JuMP.Model, bp::BilevelProblem; upperlevel=true, resvar=false)
    (_, x, v, λ, σ, s, upperfeas, lowerfeas, kkt) = high_point_relaxation(m, bp, upperlevel=upperlevel, resvar=resvar)
    @constraint(m, kkt2_bounds[i=1:bp.ml],
        [λ[i], s[i]] in MOI.SOS1([0.4,0.6])
    )
    @constraint(m, kkt2_var[j=1:bp.nl],
        [v[j], σ[j]] in MOI.SOS1([0.4,0.6])
    )
    return (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds)
end

"""
    high_point_relaxation(m::JuMP.Model, bp::BilevelProblem)

Build the relaxed high-point relaxation model, along with dual feasibility (no complementarity constraint).
This problem is purely linear & continuous.
"""
function high_point_relaxation(m::JuMP.Model, bp::BilevelProblem; upperlevel = true, resvar=false)
    @variable(m, x[1:bp.nu] >= 0)
    @variable(m, v[1:bp.nl] >= 0)
    @variable(m, σ[1:bp.nl] >= 0)
    @variable(m, λ[1:bp.ml] >= 0)
    @variable(m, s[1:bp.ml] >= 0)
    if upperlevel
        @constraint(m, bp.G * x + bp.H * v .<= bp.q)
    end
    if resvar
        @variable(m, r[i=1:bp.ml])
        @constraint(m, bp.b - bp.A * x .== r)
        @constraint(m, lowerfeas, bp.B * v  .+ s .== r)
    else
        @constraint(m, lowerfeas, bp.A * x + bp.B * v  .+ s .== bp.b)
    end
    @constraint(m, kkt, bp.d .+ bp.B' * λ .- σ .== 0)
    return (m, x, v, λ, σ, s, nothing, lowerfeas, kkt)
end

"""
Attach the upper-level objective to the model
"""
function add_upper_level(m::JuMP.Model, bp::BilevelProblem, x, v)
    @objective(m, Min, dot(bp.cx, x) + dot(bp.cy, v))
    return nothing
end

"""
    solve_near_optimal(m::JuMP.Model, bp::BilevelProblem, ::LazyExtended, δ::Real; optimizer, poly_lib)
"""
function solve_near_optimal(m::JuMP.Model, bp::BilevelProblem, ::LazyExtended, δ::Real; optimizer, poly_lib, verbose=false)
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds) = bilevel_optimality(m, bp, upperlevel=true)
    add_upper_level(m, bp, x, v)
    # early termination if relaxed or bilevel optimistic infeasible
    optimize!(m)
    st = JuMP.termination_status(m)
    w = [VariableRef[] for _ in 1:bp.mu]
    problem_ref = (model = m, x = x, v = v, λ = λ, σ = σ, s = s, w = w)
    if st != MOI.OPTIMAL && st != MOI.DUAL_INFEASIBLE
        return (Symbol(st), problem_ref)
    end
    unexplored_subproblems = BitSet(1:bp.mu)
    valid_subproblems = BitSet()
    res = bp.b - bp.A * x
    while length(unexplored_subproblems) > 0
        optimize!(m)
        empty!(valid_subproblems)
        k = pop!(unexplored_subproblems)
        subresult = dual_polyhedron(bp.B, bp.d, bp.H, k, x, v, bp.A, bp.G, bp.q, bp.b, δ; poly_lib=poly_lib, optimizer=optimizer)
        if subresult[1] == :INFEASIBLE
            return (:INFEASIBLE,)
        elseif subresult[1] == :OPTIMAL
            push!(valid_subproblems, k)
        else subresult[1] == :VERTICES
            vertex_list = subresult[2]
            w[k] = @variable(m, [l=1:length(vertex_list)], Bin, base_name = "w_$k")
            @constraint(m, sum(w[k]) >= 1)
            for l in eachindex(vertex_list)
                (α, β) = vertex_list[l]
                @constraint(m, [w[k][l], dot(α, res) + β * dot(bp.d, v) + dot(bp.G[k,:], x)]
                    in MOI.IndicatorSet{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(bp.q[k] - β * δ))
                )
            end
            optimize!(m)
            if st != MOI.OPTIMAL && st != MOI.DUAL_INFEASIBLE
                return (Symbol(st), problem_ref)
            end
            # put the optimal back in the subproblem pool
            for v in valid_subproblems
                push!(unexplored_subproblems, v)
            end
        end
    end
    return (:OPTIMAL, problem_ref)
end

"""
    solve_near_optimal(m::JuMP.Model, bp::BilevelProblem, ::LazyBatched, δ::Real; optimizer, poly_lib)
"""
function solve_near_optimal(m::JuMP.Model, bp::BilevelProblem, lb::LazyBatched, δ::Real; optimizer, poly_lib, verbose=false)
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds) = bilevel_optimality(m, bp, upperlevel=true)
    add_upper_level(m, bp, x, v)
    # early termination if relaxed or bilevel optimistic infeasible
    optimize!(m)
    st = JuMP.termination_status(m)
    w = [VariableRef[] for _ in 1:bp.mu]
    problem_ref = (model = m, x = x, v = v, λ = λ, σ = σ, s = s, w = w)
    if st != MOI.OPTIMAL && st != MOI.DUAL_INFEASIBLE
        return (Symbol(st), problem_ref)
    end
    unexplored_subproblems = BitSet(1:bp.mu)
    optimize!(m)
    valid_subproblems = BitSet()
    current_batch = BitSet()
    Γ = lb.batch_size
    while length(unexplored_subproblems) > 0
        verbose && @info "new batch, $(length(unexplored_subproblems)) unexplored subproblems"
        empty!(current_batch)
        while length(unexplored_subproblems) > 0 && length(current_batch) < Γ
            k = pop!(unexplored_subproblems)
            verbose && @info "subproblem $k"
            optimize!(m)
            subresult = dual_polyhedron(bp.B, bp.d, bp.H, k, x, v, bp.A, bp.G, bp.q, bp.b, δ; poly_lib=poly_lib, optimizer=optimizer)
            if subresult[1] == :INFEASIBLE
                verbose && @info "infeasible subproblem"
                return (:INFEASIBLE,)
            elseif subresult[1] == :OPTIMAL
                push!(valid_subproblems, k)
                verbose && @info "$k optimal"
            else subresult[1] == :VERTICES
                push!(current_batch, k)
                vertex_list = subresult[2]
                @info "Adding $(length(vertex_list)) vertices at subproblem $k"
                w[k] = @variable(m, [l=eachindex(vertex_list)], Bin, base_name = "w_$k")
                @constraint(m, sum(w[k]) >= 1)
                res = bp.b - bp.A * x
                for l in eachindex(vertex_list)
                    (α, β) = vertex_list[l]
                    @constraint(m, [w[k][l], dot(α, res) + β * dot(bp.d, v) + dot(bp.G[k,:], x)]
                        in MOI.IndicatorSet{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(bp.q[k] - β * δ))
                    )
                end
                for vsp in valid_subproblems
                    push!(unexplored_subproblems, vsp)
                end
                empty!(valid_subproblems)
            end
        end
        optimize!(m)
        st = JuMP.termination_status(m)
        if st != MOI.OPTIMAL && st != MOI.DUAL_INFEASIBLE
            return (Symbol(st), problem_ref)
        end
    end
    return (:OPTIMAL, problem_ref)
end


"""
    solve_near_optimal(m::JuMP.Model, bp::BilevelProblem, ::BilinearTerms, δ::Real; verbose=false)
"""
function solve_near_optimal(m::JuMP.Model, bp::BilevelProblem, ::BilinearTerms, δ::Real; verbose=false)
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds) = bilevel_optimality(m, bp, upperlevel=true)
    add_upper_level(m, bp, x, v)
    @variable(m, α[1:bp.mu, 1:bp.ml] >= 0)
    @variable(m, β[1:bp.mu] >= 0)
    @constraint(m, dual_polyhedron_feasible[k=1:bp.mu],
        bp.B' * α[k,:] .+ β[k] * bp.d .≥ bp.H[k,:]
    )
    @constraint(m, bilinear_cons[k=1:bp.mu],
        α[k,:] ⋅ (bp.b - bp.A * x) .+ β[k] * (bp.d ⋅ v + δ) .≤ bp.q[k] - bp.G[k,:] ⋅ x
    )
    return (model = m, x = x, v = v, λ = λ, σ = σ, s = s, α = α, β = β)
end
