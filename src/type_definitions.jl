
"""
Data specifying a bilevel linear-linear problem
"""
struct BilevelProblem
    cx::Vector{Float64}
    cy::Vector{Float64}
    G::Matrix{Float64}
    H::Matrix{Float64}
    q::Vector{Float64}
    d::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    b::Vector{Float64}
    nu::Int
    nl::Int
    mu::Int
    ml::Int
    function BilevelProblem(cx, cy, G, H, q, d, A, B, b)
        nu = length(cx)
        nl = length(cy)
        nl == length(d) || DimensionMismatch("Objectives")
        mu = length(q)
        ml = length(b)
        size(A) == (ml,nu) && size(B) == (ml,nl) || DimensionMismatch("Lower constraints")
        size(G) == (mu,nu) && size(H) == (mu,nl) || DimensionMismatch("Higher constraints")
        new(cx,cy,G,H,q,d,A,B,b,nu,nl,mu,ml)
    end
end

abstract type NearOptimalMethod end

"""
    ExtendedMethod

Solution method for the near-optimal problem using the extended formulation
"""
struct ExtendedMethod <: NearOptimalMethod end

"""
    LazyExtended

Solve the model and lazily verify which sub-problems must be added
"""
struct LazyExtended <: NearOptimalMethod end


"""
    BilinearTerms

Solution approach for the near-optimal problem using the bilinear formulation
"""
struct BilinearTerms <: NearOptimalMethod end

"""
    LazyBatched(batch_size::Int)

Solve the model with the lazy subproblem expansion and maximum batch size `batch_size`.
"""
struct LazyBatched <: NearOptimalMethod
    batch_size::Int
end
