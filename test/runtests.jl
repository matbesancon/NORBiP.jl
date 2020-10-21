using NORBiP
using Test

import SCIP
using JuMP
using LinearAlgebra
import Polyhedra
import CDDLib

test_bp() = NORBiP.BilevelProblem(
    [1.,0.], [0.],
    [-.1 0.], -ones(1,1), [-1.],
    [-1.],
    [-.1 0.], ones(1,1), ones(1,)
)

function test_bp2()
    G = ones(2,1); G[1,1] = -1
    H = 2 .* ones(2,1); H[1,1] = 4
    A = zeros(2,1) .+ [-2, 5]
    q = [11.0, 13.0]
    cx = [1.0]
    cy = [-10.0]
    B = zeros(2,1) .+ [-1, -4]
    b = [-5.0, 30.0]
    d = [1.0]
    return NORBiP.BilevelProblem(cx, cy, G, H, q, d, A, B, b)
end

@testset "Basic tests model 1" begin
    bp = test_bp()
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (m, x, v) = NORBiP.high_point_relaxation(m, bp)
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 0.0
    m2 = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (_, x, v) = NORBiP.bilevel_optimality(m2, bp)
    optimize!(m2)
    @test JuMP.termination_status(m2) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 0.0
    @test JuMP.value(v[1]) ≈ 1.0
end

@testset "Basic tests model 2" begin
    bp = test_bp2()
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (m, x, v) = NORBiP.high_point_relaxation(m, bp)
    NORBiP.add_upper_level(m, bp, x, v)
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 5.0
    @test JuMP.value(v[1]) ≈ 4.0
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (_, x, v) = NORBiP.bilevel_optimality(m, bp)
    @constraint(m, bp.G * x .+ bp.H .<= bp.q)
    @objective(m, Min, dot(bp.cx, x) + dot(bp.cy, v))
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.0
    @test JuMP.value(v[1]) ≈ 3.0
    vertex_list = [
        [(zeros(2,), 4.0)], #k = 1
        [(zeros(2,), 2.0)], #k = 2
    ]
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds, w) = NORBiP.solve_near_optimal(m, bp, NORBiP.ExtendedMethod(), 0.5, vertex_list)
    NORBiP.add_upper_level(m, bp, x, v)
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.222 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 2.555 atol = 10^-3
    empty!(m)
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds, w) = NORBiP.solve_near_optimal(m, bp, NORBiP.ExtendedMethod(), 1.0, vertex_list)
    NORBiP.add_upper_level(m, bp, x, v)
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.444 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 2.111 atol = 10^-3
    empty!(m)
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds, w) = NORBiP.solve_near_optimal(m, bp, NORBiP.ExtendedMethod(), 1.0, vertex_list, resvar=true)
    NORBiP.add_upper_level(m, bp, x, v)
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.444 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 2.111 atol = 10^-3
end

@testset "Tests model 2 with lazy constraints" begin
    bp = test_bp2()
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (st, problem_ref) = NORBiP.solve_near_optimal(m, bp, NORBiP.LazyExtended(), 0.5, optimizer = () -> SCIP.Optimizer(display_verblevel = 0), lib = CDDLib.Library(:exact))
    (x, v, w) = (problem_ref[:x], problem_ref[:v], problem_ref[:w])
    @test JuMP.termination_status(problem_ref[:model]) == MOI.OPTIMAL
    @test st === :OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.222 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 2.555 atol = 10^-3
    for k in eachindex(w)
        if !isempty(w[k])
            @test sum(JuMP.value.(w[k])) == 1
        end
    end
end

@testset "Tests model 2 with lazy constraints & batch size 1 and 5" begin
    bp = test_bp2()
    for Γ in (1, 2, 3, 10)
        m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
        (st, problem_ref) = NORBiP.solve_near_optimal(m, bp, NORBiP.LazyBatched(Γ), 0.5, optimizer = () -> SCIP.Optimizer(display_verblevel = 0), lib = CDDLib.Library(:exact))
        (x, v, w) = (problem_ref[:x], problem_ref[:v], problem_ref[:w])
        @test JuMP.termination_status(problem_ref[:model]) == MOI.OPTIMAL
        @test st === :OPTIMAL
        @test JuMP.value(x[1]) ≈ 1.222 atol = 10^-3
        @test JuMP.value(v[1]) ≈ 2.555 atol = 10^-3
        for k in eachindex(w)
            if !isempty(w[k])
                @test sum(JuMP.value.(w[k])) == 1
            end
        end
    end
end

@testset "Bound computations" begin
    vertices = [
        ([1,2],3),
        ([-1,-1],4),
    ]
    (α_bounds, β_bounds) = NORBiP.find_hypercube(vertices)
    @test α_bounds[1] == (-1,1)
    @test α_bounds[2] == (-1,2)
    @test β_bounds == (3,4)
end

@testset "Single-vertex heuristic" begin
    bp = test_bp2()
    for bsize in (1, 2, 5, 10)
        res = NORBiP.solve_near_optimal(bp, NORBiP.SingleVertexHeuristic(bsize), 0.5, () -> SCIP.Optimizer(display_verblevel = 0))
        @test length(res) == 5
        (st, model, x, v, C) = res
        @test st == MOI.OPTIMAL
        @test JuMP.value(x[1]) ≈ 1.222 atol = 10^-3
        @test JuMP.value(v[1]) ≈ 2.555 atol = 10^-3
    end
end

@testset "Randomized single-vertex heuristic" begin
    bp = test_bp2()
    for bsize in (1, 2, 5, 10)
        res = NORBiP.solve_near_optimal(bp, NORBiP.RandomizedSingleVertexHeuristic(bsize), 0.5, () -> SCIP.Optimizer(display_verblevel = 0))
        @test length(res) == 5
        (st, model, x, v, C) = res
        @test st == MOI.OPTIMAL
        @test JuMP.value(x[1]) ≈ 1.222 atol = 10^-3
        @test JuMP.value(v[1]) ≈ 2.555 atol = 10^-3
    end
end

@testset "Tests models with bilinear terms" begin
    bp = test_bp2()
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    problem_ref = NORBiP.solve_near_optimal(m, bp, NORBiP.BilinearTerms(), 0.5)
    x, v = (problem_ref[:x], problem_ref[:v])
    optimize!(m)
    @test JuMP.termination_status(problem_ref[:model]) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.222 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 2.555 atol = 10^-3

    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    problem_ref = NORBiP.solve_near_optimal(m, bp, NORBiP.BilinearTerms(), 0.0)
    x, v = (problem_ref[:x], problem_ref[:v])
    optimize!(m)
    @test JuMP.termination_status(problem_ref[:model]) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.0 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 3.0 atol = 10^-3

    bp = test_bp()
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    problem_ref = NORBiP.solve_near_optimal(m, bp, NORBiP.BilinearTerms(), 0.1)
    x, v = (problem_ref[:x], problem_ref[:v])
    optimize!(m)
    @test JuMP.termination_status(problem_ref[:model]) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 0.5 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 1.05 atol = 10^-3
end

@testset "Adversarial test model 2" begin
    bp = test_bp2()
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (_, x, v) = NORBiP.bilevel_optimality(m, bp)
    NORBiP.add_upper_level(m, bp, x, v)
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.0
    @test JuMP.value(v[1]) ≈ 3.0
    ks = NORBiP.near_optimal_robust_violated(bp, JuMP.value.(x), JuMP.value.(v), 0.5; optimizer=SCIP.Optimizer)
    @test length(ks) == 1
end

@testset "Valid inequality" begin
    # Basic tests model 2
    bp = test_bp2()
    vertex_list = [
        [(zeros(2,), 4.0)], #k = 1
        [(zeros(2,), 2.0)], #k = 2
    ]
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds, w) = NORBiP.solve_near_optimal(m, bp, NORBiP.ExtendedMethod(), 0.5, vertex_list)
    NORBiP.add_upper_level(m, bp, x, v)
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.222 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 2.555 atol = 10^-3
    obj1 = objective_value(m)
    NORBiP.add_upper_level(m, bp, x, v)
    optimize!(m)
    @test obj1 ≈ objective_value(m) atol = 10^-3
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.222 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 2.555 atol = 10^-3
    (st, Abounds) = NORBiP.compute_bounds(bp; optimizer=() -> SCIP.Optimizer(display_verblevel = 0))
    @test st == MOI.OPTIMAL
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds, w) = NORBiP.solve_near_optimal(m, bp, NORBiP.ExtendedMethod(), 0.5, vertex_list)
    NORBiP.add_upper_level(m, bp, x, v)
    NORBiP.add_bound_inequalities(m, bp, x, v, λ, Abounds)
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test obj1 ≈ objective_value(m) atol = 10^-3
    @test JuMP.value(x[1]) ≈ 1.222 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 2.555 atol = 10^-3
    m = Model(() -> SCIP.Optimizer(display_verblevel = 0))
    (m, x, v, λ, σ, s, upperfeas, lowerfeas, kkt, kkt2_var, kkt2_bounds, w) = NORBiP.solve_near_optimal(m, bp, NORBiP.ExtendedMethod(), 1.0, vertex_list)
    NORBiP.add_upper_level(m, bp, x, v)
    NORBiP.add_bound_inequalities(m, bp, x, v, λ, Abounds)
    optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x[1]) ≈ 1.444 atol = 10^-3
    @test JuMP.value(v[1]) ≈ 2.111 atol = 10^-3
end

include("test_inf.jl")
