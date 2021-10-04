##### Analytic Method #####

function compute_first_partial(j, ODE, x0, p, tspan, epsilon; kwargs...)
    # Solve perturbed problem.
    perturbation = epsilon * im
    sol = solve_perturbation(j, ODE, x0, p, tspan, perturbation; kwargs...)

    # Extract partial derivative and write to output.
    dj = @. imag(sol) / epsilon

    return dj
end

function analytic_method(::Order1, ODE::Function, x0, params, tspan, epsilon; kwargs...)
    # Get problem size info.
    num_params = length(params)
    num_vars = length(x0)

    # Solve original problem.
    sol = solve_wrapper(ODE, x0, params, tspan; kwargs...)

    # Allocate output based on dimension info from ODE solution.
    solution = Dict{Int, Matrix{Float64}}(
        i => zeros(length(sol), 1+num_params) for i in 1:num_vars)
    record_data!(solution, sol, 1, eachindex(x0))

    # Copy x0 and params as complex arrays.
    p_tmp = complex(params)
    x0_tmp = complex(x0)

    # Solve for each partial derivative.
    for j in 1:num_params
        dj = compute_first_partial(j, ODE, x0_tmp, p_tmp, tspan, epsilon; kwargs...)
        record_data!(solution, dj, 1+j, eachindex(x0))
    end

    return solution
end

function analytic_method_multi(::Order1, ODE::Function, x0, params, tspan, epsilon; kwargs...)
    # Get problem size info.
    num_params = length(params)
    num_vars = length(x0)

    # Solve original problem.
    sol = solve_wrapper(ODE, x0, params, tspan; kwargs...)

    # Allocate output based on dimension info from ODE solution.
    solution = Dict{Int, Matrix{Float64}}(
        i => zeros(length(sol), 1+num_params) for i in 1:num_vars)
    record_data!(solution, sol, 1, eachindex(x0))

    # Copy x0 and params as complex arrays.
    p_local = [complex(params) for thread in 1:Threads.nthreads()]
    x0_tmp = complex(x0)

    # Limit number of BLAS threads as this can interfere with the parallel workload.
    BLAS_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    # Solve for each partial derivative. Use param array assigned to thread.
    try
        @batch for j in 1:num_params
            id = Threads.threadid()
            dj = compute_first_partial(j, ODE, x0_tmp, p_local[id], tspan, epsilon; kwargs...)
            record_data!(solution, dj, 1+j, eachindex(x0))
        end
    finally
        # Restore previous global state.
        BLAS.set_num_threads(BLAS_threads)
    end

    return solution
end

##### DES Wrapper #####

function DES(::Order1, ODE::Function, x0, params, tspan; alg=nothing, kwargs...)
    # Solve the sensitivity problem.
    problem = ODEForwardSensitivityProblem(ODE, x0, tspan, params)
    t_end = floor(Int64, tspan[2])
    if alg isa Nothing
        sol = solve(problem; saveat=0:t_end, kwargs...)
    else
        sol = solve(problem, alg; saveat=0:t_end, kwargs...)
    end

    # Save data in the same format as analytic_method.
    solution = Dict{Int, Matrix{Float64}}(
        i => zeros(length(sol), 1+length(params)) for i in 1:length(x0))
    u, du = extract_local_sensitivities(sol)
    record_data!(solution, u, 1, eachindex(x0)) # solution
    foreach(j -> record_data!(solution, du[j], 1+j, eachindex(x0)), eachindex(params)) # partials

    return solution
end

#wrapper for the Forward Diff jacobian method
function FD(::Order1, ODE, x0, params, tspan; chunksize = nothing, multi = false, alg=nothing, kwargs...)
    # Solve the sensitivity problem.
    f = function (params)
        problem = ODEProblem(ODE, x0, tspan, params)
        t_end = floor(Int64, tspan[2])
        if alg isa Nothing
            solve(problem; saveat=0:t_end, kwargs...)
        else
            solve(problem, alg; saveat=0:t_end, kwargs...)
        end
    end
    sol = f(params)

    if isnothing(chunksize)
        chunksize = length(x0)
    end
    if multi
        dusol = Matrix(zeros((Int(tspan[2])+1) * length(x0),length(params)))
        threaded_jacobian!(p->Array(f(p)), dusol, params, ForwardDiff.Chunk(chunksize))
    else
        cfg = JacobianConfig(nothing, params, Chunk{chunksize}())
        dusol = ForwardDiff.jacobian(p->Array(f(p)), params, cfg)
    end

    # Save data in the same format as analytic_method.
    solution = Dict{Int, Matrix{Float64}}(
        i => zeros(length(sol), 1+length(params)) for i in 1:length(x0))
    record_data!(solution, sol, 1, eachindex(x0)) # solution
    for i in eachindex(x0), j in eachindex(params) # record solution for each species
#         @views solution[i][:,j+1] = dusol[i : length(x0) : end, j]
        @views copyto!(solution[i][:,j+1], dusol[i : length(x0) : end, j])
    end
    return solution
end

##### Extrapolation #####

function predict(::Order1, order1_partials, perturb)
    # Infer problem size.
    num_vars = length(order1_partials)
    num_times = size(order1_partials[1], 1)
    num_params = size(order1_partials[1], 2) - 1

    # Output has the same structure as order1_partials
    predictions = Dict{Int,Matrix{Float64}}(i => zeros(num_times, 1+num_params) for i in 1:num_vars)

    for (var, arr) in order1_partials # for each compartment...
        var_t = @view arr[:, 1] # vector
        dvar_dp = @view arr[:, 2:end] # matrix
        @views predictions[var][:, 1] .= var_t # original
        @views predictions[var][:, 2:end] .= var_t .+ (perturb' .* dvar_dp) # prediction
    end

    return predictions
end


#Polyester Code Update (can use Polyester.jl when that gets released and the threaded_jacobian! method becomes public)
function cld_fast(a::A,b::B) where {A,B}
    T = promote_type(A,B)
    cld_fast(a%T,b%T)
end
function cld_fast(n::T, d::T) where {T}
    x = Base.udiv_int(n, d)
    x += n != d*x
end
function evaluate_jacobian_chunks!(f::F, (Δx,x), start, stop, ::ForwardDiff.Chunk{C}) where {F,C}
    cfg = ForwardDiff.JacobianConfig(f, x, ForwardDiff.Chunk{C}(), nothing)

    # figure out loop bounds
    N = length(x)
    last_stop = cld_fast(N, C)
    is_last = last_stop == stop
    stop -= is_last

    # seed work arrays
    xdual = cfg.duals
    ForwardDiff.seed!(xdual, x)
    seeds = cfg.seeds

    # handle intermediate chunks
    for c ∈ start:stop
        # compute xdual
        i = (c-1) * C + 1
        ForwardDiff.seed!(xdual, x, i, seeds)
        
        # compute ydual
        ydual = f(xdual)

        # extract part of the Jacobian
        Δx_reshaped = ForwardDiff.reshape_jacobian(Δx, ydual, xdual)
        ForwardDiff.extract_jacobian_chunk!(Nothing, Δx_reshaped, ydual, i, C)
        ForwardDiff.seed!(xdual, x, i)
    end

    # handle the last chunk
    if is_last
        lastchunksize = C + N - last_stop*C
        lastchunkindex = N - lastchunksize + 1

        # compute xdual
        ForwardDiff.seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        
        # compute ydual
        _ydual = f(xdual)
        
        # extract part of the Jacobian
        _Δx_reshaped = ForwardDiff.reshape_jacobian(Δx, _ydual, xdual)
        ForwardDiff.extract_jacobian_chunk!(Nothing, _Δx_reshaped, _ydual, lastchunkindex, lastchunksize)
    end
end
function threaded_jacobian!(f::F, Δx::AbstractArray, x::AbstractArray, ::ForwardDiff.Chunk{C}) where {F,C}
    N = length(x)
    d = cld_fast(N, C)
    batch((d,min(d,num_threads())), Δx, x) do Δxx,start,stop
        evaluate_jacobian_chunks!(f, Δxx, start, stop, ForwardDiff.Chunk{C}())
    end
    return Δx
end
