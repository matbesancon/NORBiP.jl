
"""
    enum_vertices(bp::BilevelProblem, poly_lib)

Computes the extreme points of the dual polyhedron described
in `bp` using the `poly_lib` polyhedral solver (compliant with `Polyhedra`).
"""
enum_vertices(bp::BilevelProblem, poly_lib) = enum_vertices(bp.B, bp.H, bp.d, poly_lib)

"""
    enum_vertices(B, H, d, poly_lib)

Computes the extreme points of the dual polyhedron described
by `(B, H, d)` using the `poly_lib` polyhedral solver.
Returns a tuple `(V, F, k)` where:
- `V` is the vector of sub-problem results, each containing a vector of extremum points `(α, β)`
- `F` is a boolean indicating feasibility
- `k` is the first encountered infeasible sub-problem if relevant, 0 otherwise
"""
function enum_vertices(B, H, d, poly_lib)
    (ml, nl) = size(B)
    (mu, _) = size(H)
    m = JuMP.Model()
    # vertices contains for each sub-problem k a Vector of exremal points (α, β)
    vertices = Vector{Vector{Tuple{Vector{Float64}, Float64}}}(undef, mu)
    for k in 1:mu
        @variable(m, α[i = 1:ml] >= 0)
        @variable(m, β >= 0)
        @constraint(m, B' * α + β .* d .>= H[k,:])
        poly = Polyhedra.polyhedron(m, poly_lib)
        vk = Polyhedra.points(Polyhedra.vrep(poly))
        if isempty(vk) # no vertices == infeasible
            return (vertices, false, k)
        end
        vertices[k] = [(Float64.(v[1:ml]), Float64(v[end])) for v in vk]
    end
    return (vertices, true, 0)
end

function dual_polyhedron(B, d, H, k::Integer, xvar, vvar, A, G, q, b, δ::Real;
                         lib::Polyhedra.Library, optimizer)
    x = JuMP.value.(xvar)
    v = JuMP.value.(vvar)
    nl = length(d)
    ml = size(B, 1)
    mu = size(H, 1)
    m = Model(optimizer)
    @variable(m, α[i=1:ml] >= 0)
    @variable(m, β >= 0)
    @constraint(m,
        cons[j=1:nl],
        sum(B[i,j] * α[i] for i in 1:ml) + d[j] * β ≥ H[k,j]
    )
    res = b - A * x
    @objective(m, Min, dot(α, res) + β * (dot(d, v) + δ))
    optimize!(m)
    # infeasible => no extreme point, the whole problem is infeasible
    if termination_status(m) == MOI.INFEASIBLE || termination_status(m) == MOI.INFEASIBLE_OR_UNBOUNDED
        return (:INFEASIBLE, nothing)
    end
    if JuMP.objective_value(m) <= q[k] - dot(G[k,:], x)
        return (:OPTIMAL, nothing)
    end
    points = Polyhedra.points(Polyhedra.vrep(Polyhedra.polyhedron(m, lib)))
    vertices = map(points) do pvec
        alpha_vec = pvec[1:end-1]
        beta = pvec[end]
        (Float64.(alpha_vec), Float64(beta))
    end
    return (:VERTICES, vertices)
end
