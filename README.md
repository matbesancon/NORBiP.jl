# NORBiP: near-optimal robust bilevel optimization

This package is based on the [JuMP](https://jump.dev) modelling framework
and defines methods for the linear-linear near-optimal robust bilevel optimization problem.
See the [pre-print](https://arxiv.org/abs/1908.04040) for motivation, notation, and formulation of the model.

# Usage

See the tests in `test/runtests.jl` for example usage of the different algorithms.

# Requirements

NORBiP requires a MILP optimizer handling indicator constraints such as SCIP, Gurobi or CPLEX.
Those can sadly not be tested in CI (yet).
