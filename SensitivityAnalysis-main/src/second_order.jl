##### Analytic Method #####

function compute_second_partial_unmixed(j, solplus, solminus, ODE, x0, p, tspan, epsilon; kwargs...)
    # Compute perturbation coefficient.
    perturbation = epsilon * exp(pi * im  / 4)

    # Compute "plus" perturbation.
    solp = solve_perturbation(j, ODE, x0, p, tspan, perturbation; kwargs...)
    for var in eachindex(x0)
        @views solplus[var] = solp[var, :]
    end

    # Compute "minus" perturbation.
    solm = solve_perturbation(j, ODE, x0, p, tspan, -perturbation; kwargs...)
    for var in eachindex(x0)
        @views solminus[var] = solm[var, :]
    end

    # Compute unmixed second derivative.
    djdj = @. imag(solplus + solminus) / (epsilon ^ 2)

    return djdj
end

function compute_second_partial_mixed(j, k, solution, solplus, solminus, ODE, x0, p, tspan, epsilon; kwargs...)
    # Compute perturbation coefficient.
    perturbation = epsilon * exp(pi * im  / 4)

    # Compute "plus" perturbation.
    solp = solve_perturbation(j, k, ODE, x0, p, tspan, perturbation; kwargs...)
    for var in eachindex(x0)
        @views solplus[var] = solp[var, :]
    end

    # Compute "minus" perturbation.
    solm = solve_perturbation(j, k, ODE, x0, p, tspan, -perturbation; kwargs...)
    for var in eachindex(x0)
        @views solminus[var] = solm[var, :]
    end

    # Compute mixed second derivative.
    jj = parse(Int, string(j, j)); djdj = solution[jj]
    kk = parse(Int, string(k, k)); dkdk = solution[kk]

    djdk = @. 1//2 * (imag(solplus + solminus) / (epsilon ^ 2) - djdj - dkdk)

    return djdk
end

function analytic_method(::Order2, ODE::Function, x0, params, tspan, epsilon; kwargs...)
    # Get problem size information.
    num_params = length(params)
    num_vars = length(x0)

    # Infer type information sol.u (array); probably not needed
    problem = ODEProblem(ODE, x0, tspan, params)
    sol = solve(problem, saveat=1.0)
    u_t = @views complex(sol[1, :])

    # Create worker arrays for computing derivatives.
    solplus = [similar(u_t) for var in 1:num_vars]
    solminus = [similar(u_t) for var in 1:num_vars]

    # Make complex-valued copies of x0 and params.
    x0_tmp = complex(x0)
    p_tmp = complex(params)

    # Allocate output for second derivatives.
    solution = Dict{Int,Vector{Vector{Float64}}}()
    for j in 1:num_params, k in 1:j
        jk = parse(Int, string(j, k))
        solution[jk] = [zeros(length(sol)) for var in 1:num_vars]
    end

    # Compute second partial derivatives.
    for j in 1:num_params
        # Compute djdj.
        djdj = compute_second_partial_unmixed(j, solplus, solminus, 
            ODE, x0_tmp, p_tmp, tspan, epsilon; kwargs...)
        
        # Record derivative in output.
        jj = parse(Int, string(j, j))
        foreach(var -> copyto!(solution[jj][var], djdj[var]), 1:num_vars)

        for k in 1:j-1
            # Compute djdk.
            djdk = compute_second_partial_mixed(j, k, solution, solplus, solminus,
                ODE, x0_tmp, p_tmp, tspan, epsilon; kwargs...)

            # Compute derivative in output.
            jk = parse(Int, string(j, k))
            foreach(var -> copyto!(solution[jk][var], djdk[var]), 1:num_vars)
        end
    end

    return solution
end

function analytic_method_multi(::Order2, ODE::Function, x0, params, tspan, epsilon; kwargs...)
    # Get problem size information.
    num_params = length(params)
    num_vars = length(x0)

    # Infer type information sol[j,:] (vector); probably not needed
    problem = ODEProblem(ODE, x0, tspan, params)
    sol = solve(problem, saveat=1.0)
    u_t = @views complex(sol[1, :])
    T = typeof(u_t)

    # Create worker arrays for computing derivatives. Need copies for each thread.
    solplus = [Array{T,1}(undef, num_vars) for thread in 1:Threads.nthreads()]
    solminus = [Array{T,1}(undef, num_vars) for thread in 1:Threads.nthreads()]
    for id in 1:Threads.nthreads()
        solplus[id] = [similar(u_t) for var in 1:num_vars]
        solminus[id] = [similar(u_t) for var in 1:num_vars]
    end

    # Make complex-valued copies of x0 and params.
    x0_tmp = complex(x0)
    p_local = [complex(params) for thread in 1:Threads.nthreads()]

    # Allocate output for second derivatives.
    solution = Dict{Int,Vector{Vector{Float64}}}()
    for j in 1:num_params, k in 1:j
        jk = parse(Int, string(j, k))
        solution[jk] = [zeros(length(sol)) for var in 1:num_vars]
    end

    # Limit number of BLAS threads as this can interfere with the parallel workload.
    BLAS_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    # First, compute all unmixed partial derivatives in parallel...
    try
        @batch for j in 1:num_params
            id = Threads.threadid()

            # Compute djdj.
            djdj = compute_second_partial_unmixed(j, solplus[id], solminus[id], 
                ODE, x0_tmp, p_local[id], tspan, epsilon; kwargs...)
            
            # Record derivative in output.
            jj = parse(Int, string(j, j))
            for var in 1:num_vars
                solution[jj][var] .= djdj[var]
            end
             # Record derivative in output.
        end

        # ... then, compute mixed partial derivatives in parallel.
        Threads.@sync begin
            for j in 1:num_params, k in 1:j-1
                Threads.@spawn begin # multi-threaded part
                    id = Threads.threadid()
                    # Compute djdk.
                    djdk = compute_second_partial_mixed(j, k, solution, solplus[id], solminus[id], 
                        ODE, x0_tmp, p_local[id], tspan, epsilon; kwargs...)

                    # Record derivative in output.
                    jk = parse(Int, string(j, k))
                    foreach(var -> copyto!(solution[jk][var], djdk[var]), 1:num_vars)
                end # multi-threaded part
            end
        end
    finally
        # Restore previous global state.
        BLAS.set_num_threads(BLAS_threads)
    end

    return solution
end

##### DES Wrapper #####

function DES(::Order2, ODE::Function, x0, params, tspan, sensealg; alg=nothing, kwargs...)
    num_params = length(params)
    num_vars = length(x0)
    t_end = floor(Int64, tspan[2])

    # Allocate output for second derivatives.
    solution = Dict{Int,Vector{Vector{Float64}}}()
    for j in 1:num_params, k in 1:j
        jk = parse(Int, string(j, k))
        solution[jk] = [zeros(length(0:t_end)) for var in 1:num_vars]
    end
    _alg = alg isa Nothing ? Vern9() : alg
    
    # Compute second derivatives for each timestep and compartment
    problem = ODEProblem(ODE, x0, tspan, params)
    for t in 0:t_end, var in 1:num_vars
        # Extract Hessian of u(t) wrt parameters.
        if sensealg isa Nothing
            H = second_order_sensitivities(u -> u[var,1], problem, _alg;
                saveat=[t], # only store the last time point
                kwargs...   # pass additional arguments
            )
        else
            H = second_order_sensitivities(u -> u[var,1], problem, _alg;
                sensealg=sensealg,
                saveat=[t], # only store the last time point
                kwargs...   # pass additional arguments
            )
        end
        
        # Record derivatives in output.
        for j in 1:num_params, k in 1:j
            jk = parse(Int, string(j, k))
            solution[jk][var][1+t] = H[j,k]
        end
    end

    return solution
end

# wrapper for the ForwardDiff Hessian method
function FD(::Order2, ODE, x0, params, tspan; alg=nothing, kwargs...)
    num_params = length(params)
    num_vars = length(x0)
    t_end = floor(Int64, tspan[2])
    
    # Allocate output for second derivatives.
    solution = Dict{Int,Vector{Vector{Float64}}}()
    for j in 1:num_params, k in 1:j
        jk = parse(Int, string(j, k))
        solution[jk] = [zeros(length(0:t_end)) for var in 1:num_vars]
    end
    
    # Solve the sensitivity problem for a specific variable at time t.
    f = function (var, t, params)
        problem = ODEProblem(ODE, x0, (0.0,t), params)
        if alg isa Nothing
            u = solve(problem; saveat=[t], save_idxs=[var], kwargs...)
        else
            u = solve(problem, alg; saveat=[t], save_idxs=[var], kwargs...)
        end
        return u[1,1]
    end

    # Setup result and config in ForwardDiff
    H = zeros(num_params, num_params)
    
    # Compute Hessian of u(t) with respect to parameters for each variable and time point.
    # Can this be parallelized?
    for t in 0:t_end, var in 1:num_vars
        # Obtain Hessian of specified variable u(t) with respect to parameters. 
        ForwardDiff.hessian!(H, p -> f(var,t,p), params)
        
        # Record derivatives in output.
        for j in 1:num_params, k in 1:j
            jk = parse(Int, string(j, k))
            solution[jk][var][1+t] = H[j,k]
        end
    end

    return solution
end

# wrapper for double ForwardDiff.jacobian
function FD2(::Order2, ODE, x0, params, tspan; chunksize = nothing, multi = false, alg=nothing, kwargs...)
    num_params = length(params)
    num_vars = length(x0)
    t_end = floor(Int64, tspan[2])
    
    # Allocate output for second derivatives.
    solution = Dict{Int,Vector{Vector{Float64}}}()
    for j in 1:num_params, k in 1:j
        jk = parse(Int, string(j, k))
        solution[jk] = [zeros(length(0:t_end)) for var in 1:num_vars]
    end
    
    f = function (params)
        problem = ODEProblem(ODE, x0, tspan, params)
        t_end = floor(Int64, tspan[2])
        if alg isa Nothing
            solve(problem; saveat=0:t_end, kwargs...)
        else
            solve(problem, alg; saveat=0:t_end, kwargs...)
        end
    end
    if isnothing(chunksize)
        chunksize = length(x0)
    end
    if multi
        Htmp = Matrix(zeros((Int(tspan[2])+1) * length(params)^2, length(x0)))
        cfg = JacobianConfig(nothing, params, Chunk{chunksize}())
        threaded_jacobian!(p->ForwardDiff.jacobian(q->Array(f(q)), p, JacobianConfig(nothing, p, Chunk{chunksize}())), Htmp, params, ForwardDiff.Chunk(chunksize))
    else
      cfg = JacobianConfig(nothing, params, Chunk{chunksize}())
      Htmp = ForwardDiff.jacobian(p -> Array(ForwardDiff.jacobian(q -> Array(f(q)), p, JacobianConfig(nothing, p, Chunk{chunksize}()))), params, cfg)
    end
    H = reshape(Htmp, num_vars, length(0:t_end), num_params, num_params)
    # Record derivatives in output.
    for k in 1:num_params, j in 1:k, var in 1:num_vars
        jk = parse(Int, string(k, j))
        #solution[jk][var][:] = H[j,:,k,var]
        copyto!(solution[jk][var], view(H, var, :, j, k))
        
    end

    return solution
end

##### Extrapolation #####

function predict(::Order2, order1_partials, order2_partials, perturb)
    num_params = size(order1_partials[1], 2) - 1

    # First do the O(Δ) prediction
    predictions = predict(Order1(), order1_partials, perturb)

    # Accumulate the O(Δ²) terms
    for (var, u) in predictions, t in axes(u, 1)
        # u is the matrix corresponding to compartment var.
        # Extrapolate perturbed trajectory for each parameter j.
        for j in 1:num_params
            order2_term = zero(eltype(u))

            # Accumulate the diagonal term
            jj = parse(Int, string(j, j))
            H_jj = order2_partials[jj][var][t]
            order2_term += 1//2 * H_jj * perturb[j]^2

            # for k in 1:j-1
            #     # Accumulate the off-diagonal terms
            #     jk = parse(Int, string(j, k))
            #     H_jk = order2_partials[jk][var][t]
            #     order2_term += H_jk * perturb[j] * perturb[k]
            # end

            # Save result; remember offset by 1
            u[t, 1+j] += order2_term
        end
    end

    return predictions
end

#Polyester Code Update (can use Polyester.jl when that gets released and the threaded_jacobian! method becomes public)
#=function cld_fast(a::A,b::B) where {A,B}
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
end=#