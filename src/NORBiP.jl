module NORBiP

using JuMP
import SCIP

import Polyhedra

using LinearAlgebra: dot, ⋅

include("type_definitions.jl")
include("dual_polyhedron.jl")
include("algorithms.jl")
include("cutting_plane.jl")
include("single_vertex_heuristic.jl")
include("adversarial.jl")
include("valid_inequalities.jl")

# type piracy for the greater good
Base.string(::NearOptimalMethod) = string(typeof(mt))
Base.string(mt::LazyBatched) = "lazy_batched_$(mt.batch_size)"
Base.string(::ExtendedMethod) = "extended"
Base.string(::LazyExtended) = "lazy"
Base.string(mt::SingleVertexHeuristic) = "single_heuristic_$(mt.batch_size)"
Base.string(mt::RandomizedSingleVertexHeuristic) = "random_sv_$(mt.batch_size)"
Base.string(::BilinearTerms) = "bilinear"

end # module
