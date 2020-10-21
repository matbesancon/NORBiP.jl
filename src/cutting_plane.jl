struct CuttingPlane end

function solve_near_optimal(m::JuMP.Model, bp::BilevelProblem, lb::CuttingPlane, δ::Real; optimizer, lib, verbose=false)
    subresult = dual_polyhedron(bp.B, bp.d, bp.H, k, x, v, bp.A, bp.G, bp.q, bp.b, δ; lib=lib, optimizer=optimizer)
    if subresult[1] == :INFEASIBLE
        return (:INFEASIBLE,)
    elseif subresult[1] == :OPTIMAL
        push!(valid_subproblems, k)
    else subresult[1] == :VERTICES
        vertex_list = subresult[2]
    end
    # TODO finish?
end

function build_initial_relaxation(m::JuMP.Model, bp::BilevelProblem, δ::Real; optimizer, lib, verbose=false)
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds) = bilevel_optimality(m, bp, upperlevel=true)
    add_upper_level(m, bp, x, v)
    # early termination if relaxed or bilevel optimistic infeasible
    optimize!(m)
    st = JuMP.termination_status(m)
    w = Vector{Vector{VariableRef}}(undef, bp.mu)
    problem_ref = (model = m, x = x, v = v, λ = λ, σ = σ, s = s, w = w)
    if st != MOI.OPTIMAL && st != MOI.DUAL_INFEASIBLE
        return (Symbol(st), problem_ref)
    end
    unexplored_subproblems = BitSet(1:bp.mu)
    valid_subproblems = BitSet()
    while length(unexplored_subproblems) > 0
        optimize!(m)
        empty!(valid_subproblems)
        k = pop!(unexplored_subproblems)
        subresult = dual_polyhedron(bp.B, bp.d, bp.H, k, x, v, bp.A, bp.G, bp.q, bp.b, δ; lib=lib, optimizer=optimizer)
        if subresult[1] == :INFEASIBLE
            return (:INFEASIBLE,)
        elseif subresult[1] == :OPTIMAL
            push!(valid_subproblems, k)
        else subresult[1] == :VERTICES
            vertex_list = subresult[2]
            initial_vertices()
            w[k] = @variable(m, [l=1:length(vertex_list)], Bin, base_name = "w_$k")
            @constraint(m, sum(w[k]) >= 1)
            for l in eachindex(vertex_list)
                (α, β) = vertex_list[l]
                @constraint(m, [w[k][l], dot(α, s) + β * dot(bp.d, v) + dot(bp.G[k,:], x)]
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

function find_hypercube(vertices::AbstractVector{<:Tuple})
    nv = length(vertices)
    (α, β) = first(vertices)
    n = length(α)
    α_bounds = [(α[i], α[i]) for i in eachindex(α)]
    β_bounds = (β, β)
    for l in 2:nv
        (α, β) = vertices[l]
        β_bounds = (min(β_bounds[1], β), max(β_bounds[2], β))
        for i in eachindex(α)
            α_bounds[i] = (min(α_bounds[i][1], α[i]), max(α_bounds[i][2], α[i]))
        end
    end
    return (α_bounds, β_bounds)
end
