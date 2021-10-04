module AnalyticSensitivity

using LinearAlgebra, BenchmarkTools, Statistics, Polyester
using QuadGK, DelimitedFiles

using DifferentialEquations
using DiffEqSensitivity
using ForwardDiff
using ForwardDiff: Chunk, JacobianConfig

# use value types for dispatch
Order1 = Val{1}
Order2 = Val{2}

export Order1, Order2

# helper function to handle recording data for compartment i along column j
# used by first order methods
function record_data!(solution, sol, j, idxs)
    for i in idxs # record solution for each species
        @views copyto!(solution[i][:,j], sol[i, :])
    end
end

# Helper function to help propogate keyword arguments to solve call
function solve_wrapper(ODE, x0, p, tspan; alg=nothing, kwargs...)
    problem = ODEProblem(ODE, x0, tspan, p)
    ts = 0:floor(Int, tspan[2])
    if alg isa Nothing
        sol = solve(problem; saveat=ts, kwargs...)
    else
        sol = solve(problem, alg; saveat=ts, kwargs...)
    end
    return sol
end

# Helper functions for solving perturbed problem.
# Assumes x0, p and perturbation are complex-valued.
# Note that perturbation is always added to parameter.
# Guarantees that p retains its original values once the function exits.
function solve_perturbation(j, ODE, x0, p, tspan, perturbation; kwargs...)
    # Perturb parameter i and solve ODE in complex plain.
    p_j = p[j]
    p[j] = p_j + perturbation
    sol = solve_wrapper(ODE, x0, p, tspan; kwargs...)

    # Reset parameter array.
    p[j] = p_j
    
    return sol
end

function solve_perturbation(j, k, ODE, x0, p, tspan, perturbation; kwargs...)
    # Perturb parameters j and k, then solve ODE in complex plain.
    p_j, p_k = p[j], p[k]
    p[j], p[k] = p_j + perturbation, p_k + perturbation
    sol = solve_wrapper(ODE, x0, p, tspan; kwargs...)

    # Reset parameter array.
    p[j], p[k] = p_j, p_k
    
    return sol
end

#=
ODE Algorithms
=#
include("first_order.jl")
include("second_order.jl")

export analytic_method, analytic_method_multi, DES, predict, FD, FD2

#=
Branching Process
=#
include("BirthDeathMigration.jl")

#=
Example ODEs
=#
include("example_models.jl")

export CARRGO, SIR, LotkaVolterra, Vaccine, MCC

#=
Benchmarking
=#
include("benchmark_code.jl")

export CARRGOVisualizations,
    SIRVisualizations,
    LotkaVolterraVisualizations,
    VaccineVisualizations,
    predictExact,
    comparePredictions,
    ODEAccuracy,
    benchmarkSIR,
    benchmarkSIRAccuracy,
    benchmarkSIRTime, 
    benchmarkMCC, 
    benchmarkCARRGO

include("Benchmarking_BirthDeathMigration.jl")

end # module
