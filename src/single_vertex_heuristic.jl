using Random

struct SingleVertexHeuristic <: NearOptimalMethod
    batch_size::Int
end

struct RandomizedSingleVertexHeuristic <: NearOptimalMethod
    batch_size::Int
end

function solve_near_optimal(bp::BilevelProblem, mt::SingleVertexHeuristic, δ::Real, optimizer; silent=true, sublimit=500)
    eta = mt.batch_size
    model = Model(optimizer)
    silent && MOI.set(model, MOI.Silent(), true)
    (model, x, v, λ, σ, s, _, _, _, _, _) = bilevel_optimality(model, bp, upperlevel = true)
    (mu, _) = size(bp.G)
    add_upper_level(model, bp, x, v)
    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL && st != MOI.DUAL_INFEASIBLE
        return (st, model, nothing, nothing)
    end
    dual_problems = build_dual_adversarial(bp, optimizer)
    for (m, _, _) in dual_problems
        optimize!(m)
        if termination_status(m) != MOI.OPTIMAL
            return (MOI.INFEASIBLE, k)
        end
    end
    C=BitSet()
    count = 1 # number of subproblems added at outer iteration
    while count > 0 # outer iteration
        k = 1
        while k in C
            k += 1
        end
        count = 0
        xv = JuMP.value.(x)
        vv = JuMP.value.(v)
        while count <= eta && k <= mu
            (m, α, β) = dual_problems[k]
            MOI.set(m, MOI.TimeLimitSec(), sublimit)
            @objective(m, Min, dot(α, bp.b - bp.A * xv) + β * (dot(bp.d, vv) + δ))
            optimize!(m)
            if termination_status(m) != MOI.OPTIMAL
                @info "Unexpected dual status $k status: $(termination_status(m))"
                return (termination_status(m), :DualError, model, k)
            end
            # new vertex to add
            if objective_value(m) > bp.q[k] - dot(bp.G[k,:], xv)
                push!(C, k)
                αv = JuMP.value.(α)
                βv = JuMP.value.(β)
                @constraint(model,
                    dot(αv, bp.b - bp.A * x) + βv * (dot(bp.d, v) + δ) <= bp.q[k] - dot(bp.G[k,:], x)
                )
                count += 1
            end
            k0 = k
            while k in C || k == k0
                k += 1
            end
        end
        optimize!(model)
        st = termination_status(model)
        if st != MOI.OPTIMAL
            return (st, :NotFound, model)
        end
    end
    return (MOI.OPTIMAL, model, x, v, C)
end

function solve_near_optimal(bp::BilevelProblem, mt::RandomizedSingleVertexHeuristic, δ::Real, optimizer; silent=true, sublimit=500, rng=Random.MersenneTwister(33))
    eta = mt.batch_size
    model = Model(optimizer)
    silent && MOI.set(model, MOI.Silent(), true)
    (model, x, v, λ, σ, s, _, _, _, _, _) = bilevel_optimality(model, bp, upperlevel = true)
    (mu, _) = size(bp.G)
    add_upper_level(model, bp, x, v)
    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL && st != MOI.DUAL_INFEASIBLE
        return (st, model, nothing, nothing)
    end
    dual_problems = build_dual_adversarial(bp, optimizer)
    for (m, _, _) in dual_problems
        optimize!(m)
        if termination_status(m) != MOI.OPTIMAL
            return (MOI.INFEASIBLE, k)
        end
    end
    C=BitSet()
    count = 1 # number of subproblems added at outer iteration
    while count > 0 # outer iteration
        korder = permute!(
            [k for k in 1:mu if k ∉ C],
            randperm(rng, mu - length(C)),
        )
        k = pop!(korder)
        count = 0
        xv = JuMP.value.(x)
        vv = JuMP.value.(v)
        while count <= eta
            @info "expanding sub-problem $k"
            (m, α, β) = dual_problems[k]
            MOI.set(m, MOI.Silent(), true)
            MOI.set(m, MOI.TimeLimitSec(), sublimit)
            @objective(m, Min, dot(α, bp.b - bp.A * xv) + β * (dot(bp.d, vv) + δ))
            optimize!(m)
            if termination_status(m) != MOI.OPTIMAL
                @info "Unexpected dual status $k status: $(termination_status(m))"
                return (termination_status(m), :DualError, model, k)
            end
            # new vertex to add
            if objective_value(m) > bp.q[k] - dot(bp.G[k,:], xv)
                push!(C, k)
                αv = JuMP.value.(α)
                βv = JuMP.value.(β)
                @constraint(model,
                    dot(αv, bp.b - bp.A * x) + βv * (dot(bp.d, v) + δ) <= bp.q[k] - dot(bp.G[k,:], x)
                )
                count += 1
            end
            if isempty(korder)
                break
            end
            k = pop!(korder)
        end
        optimize!(model)
        st = termination_status(model)
        if st != MOI.OPTIMAL
            return (st, :NotFound, model)
        end
    end
    return (MOI.OPTIMAL, model, x, v, C)
end

function build_dual_adversarial(bp, optimizer)
    (mu, nl) = size(bp.H)
    ml = size(bp.B, 1)
    return map(1:mu) do k
        m = Model(optimizer)
        @variable(m, α[i=1:ml] >= 0)
        @variable(m, β >= 0)
        @constraint(m,
            cons[j=1:nl],
            sum(bp.B[i,j] * α[i] for i in 1:ml) + bp.d[j] * β ≥ bp.H[k,j]
        )
        (m, α, β)
    end
end
