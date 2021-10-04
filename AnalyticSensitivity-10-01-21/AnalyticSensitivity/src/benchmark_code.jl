#=
Visualizing Derivative-Based Trajectory Predictions

This section contains methods for creating graphs or trajectories of systems
based on the predictions created from derivatives.  These methods also
calculate the distance between the method results and the actual trajectory of
the system under perturbed parameters.

The systems included in this section include:
     SIR
     CARRGO
     Lotka-Volterra
=#

"""
    CARRGOVisualizations(t_end::Float64, epsilon::Float64)::Dict

Using the CARRGO system of ODEs, compare the predicted trajectories under
perturbed parameters using prediction from various methods of calculating the
derivatives.

The epsilon parameter is the amount to perturb the parameter into the complex
plane.

The output is a dictionary.  Trajectory plots can be accessed using the key
"<var_name>d<param_num>plot".  Total differences between predicted and
calculated trajectories can be accessed via the key
"<var_name>d<param_num>diff<method>", with the methods being "1" (analytic first
derivative), "1j" (julia first derivative), "2" (analytic second derivative), and
"2j" (julia second derivative).

# Example
```julia-repl
julia> CARRGOVisualizations(10.0, 1e-6)
Dict{String,Any} with 50 entries:
  "cancer cellsdp1diff1j" => 635.765
  "immune cellsdp5diff2"  => 0.000145476
  "cancer cellsdp2plot"   => Plot{Plots.GRBackend() n=6}
  "immune cellsdp1diff2j" => 0.000145057
  "cancer cellsdp5diff2j" => 0.00644305
  "cancer cellsdp1diff1"  => 0.00594098
  "immune cellsdp4diff2j" => 2.26298e-7
  "immune cellsdp5plot"   => Plot{Plots.GRBackend() n=6}
  "immune cellsdp4diff1"  => 0.000182491
  "immune cellsdp4diff1j" => 2.20962e-9
  "immune cellsdp3diff2j" => 3.78589e-5
  "cancer cellsdp1diff2j" => 635.765
  "cancer cellsdp3diff2j" => 635.804
  "immune cellsdp2diff1"  => 4.07261e-7
  ⋮                       => ⋮
```
"""
function CARRGOVisualizations(t_end::Float64, epsilon::Float64; kwargs...)::Dict
    x0 = [1.25e4, 6.25e2] #initial values
    params = [6.0e-9, 3.0e-11, 1.0e-6, 6.0e-2, 1.0e9] #parameter values
    tspan = (0.0, t_end) #time to run trajectories
    del_param = 0.01 #proportion to change each parameter for predictions
    x0_labels = ["cancer cells", "immune cells"] #names of each compartment
    #create the plots & comparisons
    visuals = comparePredictions(CARRGO, x0, params, epsilon, tspan, del_param, x0_labels; kwargs...)
    return visuals
end

"""
    SIRVisualizations(t_end::Float64, population::Float64, proportion_infected::Float64, epsilon::Float64)::Dict

Using the SIR system of ODEs, compare the predicted trajectories under
perturbed parameters using prediction from various methods of calculating the
derivatives.

The output is a dictionary.  Trajectory plots can be accessed using the key
"<var_name>d<param_num>plot".  Total differences between predicted and
calculated trajectories can be accessed via the key
"<var_name>d<param_num>diff<method>", with the methods being "1" (analytic first
derivative), "1j" (julia first derivative), "2" (analytic second derivative), and
"2j" (julia second derivative).

# Example
```julia-repl
julia> SIRVisualizations(10.0, 10.0, 0.1, 1e-6)
Dict{String,Any} with 45 entries:
  "Idp1plot"   => Plot{Plots.GRBackend() n=6}
  "Rdp2diff2j" => 0.296795
  "Sdp1diff2"  => 4.31299e-8
  "Sdp3diff2"  => 0.00424312
  "Sdp3diff2j" => 9.82376e-5
  "Sdp1plot"   => Plot{Plots.GRBackend() n=6}
  "Rdp1plot"   => Plot{Plots.GRBackend() n=6}
  "Idp2diff1"  => 0.00431419
  "Rdp1diff1j" => 0.181552
  "Sdp3diff1j" => 6.43101e-5
  "Idp3diff1"  => 0.00117835
  "Rdp1diff1"  => 4.48385e-6
  "Sdp1diff2j" => 0.00195093
  "Idp3diff2j" => 5.10772e-5
  ⋮            => ⋮
```
"""
function SIRVisualizations(t_end::Float64, population::Float64, proportion_infected::Float64, epsilon::Float64; kwargs...)::Dict
    number_infected = population * proportion_infected
    x0 = [population - number_infected, number_infected, 0.0] #initial values
    params = [0.105, 0.12, population] #parameter values
    tspan = (0.0, t_end) #timesteps to run trajectories
    del_param = 0.01 #proportion to change parameters for predictions
    x0_labels = ["S", "I", "R"] #names of compartments
    #create plots and comparisons
    visuals = comparePredictions(SIR, x0, params, epsilon, tspan, del_param, x0_labels; kwargs...)
    return visuals
end

"""
    LotkaVolterraVisualizations(t_end::Float64, population::Float64, predator_proportion::Float64)::Dict

Using the Lotka-Volterra system of ODEs, compare the predicted trajectories under
perturbed parameters using prediction from various methods of calculating the
derivatives.

The output is a dictionary.  Trajectory plots can be accessed using the key
"<var_name>d<param_num>plot".  Total differences between predicted and
calculated trajectories can be accessed via the key
"<var_name>d<param_num>diff<method>", with the methods being "1" (analytic first
derivative), "1j" (julia first derivative), "2" (analytic second derivative), and
"2j" (julia second derivative).

# Example
```julia-repl
julia> LotkaVolterraVisualizations(10.0, 10.0, 0.1)
Dict{String,Any} with 40 entries:
  "predatordp1diff1j" => 0.403209
  "predatordp3diff2j" => 0.613255
  "predatordp2plot"   => Plot{Plots.GRBackend() n=6}
  "preydp3diff1"      => 0.371894
  "preydp2diff2"      => 0.322662
  "preydp1diff2"      => 0.00553236
  "preydp4plot"       => Plot{Plots.GRBackend() n=6}
  "preydp4diff1"      => 0.578362
  "preydp1diff1j"     => 0.883263
  "preydp4diff2j"     => 1.67853
  "predatordp1diff1"  => 0.00304481
  "predatordp2diff1"  => 0.250598
  "preydp4diff2"      => 0.573592
  "preydp2diff1j"     => 1.39653
  ⋮                   => ⋮
```
"""
function LotkaVolterraVisualizations(t_end::Float64, population::Float64, predator_proportion::Float64; kwargs...)::Dict
    num_predator = predator_proportion * population
    x0 = [population - num_predator, num_predator] #initial values
    params = [1.1, 0.4, 0.1,0.4] #parameter values
    epsilon = 1e-6 #amount to perturb into the complex plane
    tspan = (0.0, t_end) #timespan to run the trajectories
    del_param = 0.01 #proportion to change parameters for predictions
    x0_labels = ["prey", "predator"] #compartment names
    #create plots and compute differences
    visuals = comparePredictions(LotkaVolterra, x0, params, epsilon, tspan, del_param, x0_labels; kwargs...)
    return visuals
end

"""
    VaccineVisualizations(t_end::Float6)::Dict

Using the vaccine system of ODEs from Alfonso Landeros, compare the predicted
trajectories under perturbed parameters using prediction from various methods of
calculating the derivatives.

The output is a dictionary.  Trajectory plots can be accessed using the key
"<var_name>d<param_num>plot".  Total differences between predicted and
calculated trajectories can be accessed via the key
"<var_name>d<param_num>diff<method>", with the methods being "1" (analytic first
derivative), "1j" (julia first derivative), "2" (analytic second derivative), and
"2j" (julia second derivative).

# Example
```julia-repl
julia> VaccineVisualizations(10.0)
Dict{String,Any} with 315 entries:
  "Rdp2diff2j"   => 0.0666533
  "Rdp4diff1j"   => 0.00699142
  "I_0dp4diff1j" => 0.00894948
  "I_0dp4plot"   => Plot{Plots.GRBackend() n=6}
  "S_0dp8diff2j" => 2.34394e-5
  "S_2dp8diff2j" => 0.00494001
  "S_0dp2plot"   => Plot{Plots.GRBackend() n=6}
  "Rdp6diff2j"   => 0.013746
  "I_1dp7plot"   => Plot{Plots.GRBackend() n=6}
  "I_0dp2plot"   => Plot{Plots.GRBackend() n=6}
  "I_1dp8diff1j" => 1.88174e-5
  "I_2dp4diff1"  => 0.00015231
  "I_1dp1diff1j" => 0.00734237
  "S_2dp8plot"   => Plot{Plots.GRBackend() n=6}
  ⋮              => ⋮
```
"""
function VaccineVisualizations(t_end::Float64; kwargs...)::Dict
    #x1 = S_0; x2 = S_1; x3 = S_2; x4 = I_0; x5 = I_1; x6 = I_2; x7 = R
    #p1 = lambda_0; p2 = nu_0, p3 = delta_1, p4 = delta_2, p5 = rho, p6 = lambda_1; p7 = nu_1; p8 = lambda_2; p9 = gamma
    x0 = [1000.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0] #initial values
    params = [0.1, 0.5, .005, .0025, .01, 0.04, 1.0, .005, .12] #parameter values
    epsilon = 1e-6 #amount to perturb into the complex plane
    tspan = (0.0, t_end) #timespan to run the trajectories
    del_param = 0.01 #proportion to change parameters for predictions
    x0_labels = ["S_0", "S_1", "S_2", "I_0", "I_1", "I_2", "R"] #compartment names
    #create plots and compute differences
    visuals = comparePredictions(Vaccine, x0, params, epsilon, tspan, del_param, x0_labels; kwargs...)
    return visuals
end

function predictExact(ODE::Function, x0::Array, params::Array, tspan::Tuple, perturb::Array; alg=nothing, kwargs...)::Array
    num_params = length(params)
    num_vars = length(x0)
    new_params = copy(params)
    solutions=Array{Any, 1}(undef, num_params)
    #calculate exact solutions after perturbing each parameter
    for param in 1:num_params
        solution = Dict{Int, Any}()
        old_param = params[param]
        new_params[param] = old_param + perturb[param]
        problem = ODEProblem(ODE, x0, tspan, new_params)
        if alg isa Nothing
            sol = solve(problem, saveat = 1.0)
        else
            sol = solve(problem, alg, saveat = 1.0)
        end
        new_params[param] = old_param
        #save each compartment
        for var = 1:num_vars
            solution[var] = transpose(hcat(real(sol.u)...))[:, var]
        end
        solutions[param]=solution
    end
    return solutions
end

"""
    comparePredictions(ODE::Function, x0::Array, params::Array, epsilon::Float64, tspan::Tuple, del_param::Float64, x0_labels::Array)::Dict

Using the provided system of ODEs, compare the predicted trajectories under
perturbed parameters using prediction from various methods of calculating the
derivatives.

The ODE input function must be in the format compatible with the
DifferentialEquations.jl package.  For an example, see the "SIR" function.

The output is a dictionary.  Trajectory plots can be accessed using the key
"<var_name>d<param_num>plot".  Total differences between predicted and
calculated trajectories can be accessed via the key
"<var_name>d<param_num>diff<method>", with the methods being "1" (analytic first
derivative), "1j" (julia first derivative), "2" (analytic second derivative), and
"2j" (julia second derivative).

# Example
```julia-repl
julia> comparePredictions(SIR, [9.0, 1.0, 0.0], [0.5, 0.5, 10.0], 1e-6, (0.0, 10.0), 0.01, ["S", "I", "R"])
Dict{String,Any} with 45 entries:
  "Idp1plot"   => Plot{Plots.GRBackend() n=6}
  "Rdp2diff2j" => 2.10445
  "Sdp1diff2"  => 9.8506e-6
  "Sdp3diff2"  => 0.092586
  "Sdp3diff2j" => 0.00839036
  "Sdp1plot"   => Plot{Plots.GRBackend() n=6}
  "Rdp1plot"   => Plot{Plots.GRBackend() n=6}
  "Idp2diff1"  => 0.0834336
  "Rdp1diff1j" => 3.71126
  "Sdp3diff1j" => 0.0139996
  "Idp3diff1"  => 0.0353413
  "Rdp1diff1"  => 0.000904999
  "Sdp1diff2j" => 0.194496
  "Idp3diff2j" => 0.0030871
  ⋮            => ⋮
```
"""
function comparePredictions(ODE::Function, x0::Array, params::Array, epsilon::Float64, tspan::Tuple, del_param::Float64, x0_labels::Array; kwargs...)::Dict
    perturb = del_param * params
    #calculate derivatives
    analytic_first = analytic_method(Order1(), ODE, x0, params, tspan, epsilon; kwargs...)
    analytic_second = analytic_method(Order2(), ODE, x0, params, tspan, epsilon; kwargs...)
    julia_first = DES(Order1(), ODE, x0, params, tspan; kwargs...)
    julia_second = DES(Order2(), ODE, x0, params, tspan; kwargs...)
    #calculate trajectories
    original = predictExact(ODE, x0, params, tspan, zeros(length(params)); kwargs...)
    exact = predictExact(ODE, x0, params, tspan, perturb; kwargs...)
    analytic_first_prediction = predict(Order1(), analytic_first, perturb)
    analytic_second_prediction = predict(Order2(), analytic_first, analytic_second, perturb)
    julia_first_prediction = predict(Order1(), julia_first, perturb)
    julia_second_prediction = predict(Order2(), julia_first, julia_second, perturb)

    all_plots = Dict{String,Any}()
    for compartment = 1:length(x0)
        for param = 1:length(params)
            #plot trajectories
            single_plot = plot(analytic_first_prediction[compartment][:, 1+param], label = "Analytic First Order")
            plot!(single_plot, analytic_second_prediction[compartment][:, 1+param], label = "Analytic Second Order")
            plot!(single_plot, original[param][compartment], label = "Original")
            plot!(single_plot, exact[param][compartment], label = "Perturbed", xlabel = "days", xlims = (tspan[1], tspan[2]), title = string(x0_labels[compartment], " using dp", param))
            plot!(single_plot, julia_first_prediction[compartment][:, 1+param], label = "Julia First Order")
            plot!(single_plot, julia_second_prediction[compartment][:, 1+param], label = "Julia Second Order")
            #compute distance between predicted and actual trajectories
            all_plots[string(x0_labels[compartment], "dp", param, "plot")] = single_plot
            all_plots[string(x0_labels[compartment], "dp", param, "diff1")] = sum(abs.(exact[param][compartment] - analytic_first_prediction[compartment][:, 1+param]))
            all_plots[string(x0_labels[compartment], "dp", param, "diff2")] = sum(abs.(exact[param][compartment] - analytic_second_prediction[param][compartment][:, 1+param]))
            all_plots[string(x0_labels[compartment], "dp", param, "diff1j")] = sum(abs.(exact[param][compartment][2:end] - julia_first_prediction[compartment][:, 1+param]))
            all_plots[string(x0_labels[compartment], "dp", param, "diff2j")] = sum(abs.(exact[param][compartment][2:end] - julia_second_prediction[compartment][:, 1+param]))
        end
    end
    return all_plots
end

#=
Debugging Functions

The following section includes methods used to debug code and double check
methods.
=#
"""
    SIRFirstDerivs(dx, x, p, t)

System of ODEs for the SIR model and the first derivatives with respect to δ.
Compatible with the methods in this code as well as DifferentialEquations.jl
"""
function SIRFirstDerivs(dx, x, p, t)
    S, I, R, dSdD, dIdD, dRdD = x
    η, δ, N = p
    dx[1] = -η * S * I / N #dS/dt
    dx[2] = η * S * I / N - δ * I #dI/dt
    dx[3] = δ * I #dR/dt
    dx[4] = -η * (S * dIdD + I * dSdD) / N #dS/dtddelta
    dx[5] = η * (S * dIdD + I * dSdD) / N - I - δ * dIdD #dI/dtddelta
    dx[6] = I + δ * dIdD #dR/dtddelta
end

"""
    SIRSecondDerivs(dx, x, p, t)

System of ODEs for SIR model as well as the first and second derivatives with
respect to the δ parameter.  Compatible with the methods in this code
as well as DifferentialEquations.jl
"""
function SIRSecondDerivs(dx::Array, x::Array, params::Array, t::Tuple)
    S, I, R, dSdD, dIdD, dRdD, dSdD2, dIdD2, dRdD2 = x
    η, δ, N = p
    dx[1] = -η * S * I / N #dS/dt
    dx[2] = η * S * I / N - δ * I #dI/dt
    dx[3] = δ * I #dR/dt
    dx[4] = -η * (S * dIdD + I * dSdD) / N #dS/dtddelta
    dx[5] = η * (S * dIdD + I * dSdD) / N - (I + δ * dIdD) #dI/dtddelta
    dx[6] = I + δ * dIdD #dR/dtddelta
    dx[7] = -η * (2 * dIdD * dSdD + S * dIdD2 + I * dSdD2) / N #dS/dtddelta^2
    dx[8] = η * (2 * dIdD * dSdD + S * dIdD2 + I * dSdD2) / N - (2 * dIdD + δ * dIdD2) #dI/dtddelta^2
    dx[9] = 2 * dIdD + δ * dIdD2 #dR/dtddelta^2
end

"""
    sinxy(x,y)

Simple test function to test derivative values.

# Example
```julia-repl
julia> sinxy(0.5, 2.0)
0.8414709848078965
```
"""
function sinxy(x,y)
    return sin(x * y)
end

"""
    firstTest(x::Number, y::Number, epsilon::Float64, test_function::Function)::Tuple

Use new methods to calculate first derivatives of a simple function for testing.

# Example
```julia-repl
julia> firstTest(0.5, 2.0, 1e-6, sinxy)
(1.0806046117369998, 0.2701511529340811)
```
"""
function firstTest(x::Number, y::Number, epsilon::Float64, test_function::Function)::Tuple
    #compute first derivatives using analytic method
    dx = imag(test_function(x + (epsilon*im), y)) / epsilon
    dy = imag(test_function(x, y + (epsilon*im))) / epsilon
    return (dx,dy)
end

"""
    secondTest(x::Number, y::Number, del::Float64, test_function::Function)

Use new methods to calculate second derivatives of a simple function for
testing.

# Example
```julia-repl
julia> secondTest(0.5, 2.0, 1e-6, sinxy)
(-3.365883939261975, -0.21036774617740367, [-0.30116867906093847])
```
"""
function secondTest(x::Number, y::Number, del::Float64, test_function::Function)
    #compute second derivatives using analytic method
    epsilon = del * exp(pi * im / 4)
    dx2 = imag(test_function(x + epsilon, y) + test_function(x - epsilon, y)) / del^2
    dy2 = imag(test_function(x, y + eps) + testFun(x, y - epsilon)) / del^2
    dxdy = .5 * [imag(test_function(x + eps, y + eps) + test_function(x - epsilon, y - epsilon)) / del^2 - (dx2 + dy2)]
    return dx2, dy2, dxdy
end

#=
Other benchmarking code for examples
=#
function euclidean_dist(x, y)
    z = x - y
    return norm(z, 2)
end

function ODEAccuracy(ODE, x0, params, sensealg, epsilon1, epsilon2, time, excludeDES2=false, excludeDES2_nonadjoint=false; kwargs...)
    change = 0.1
    perturb = change * params
    
    analytic2_accuracy = 0.0
    analytic2multi_accuracy = 0.0
    FD2multi_accuracy = 0.0
    FD2chunk1multi_accuracy = 0.0
    julia2_accuracy = 0.0
    nonadjoint2_accuracy = 0.0
    analytic1_accuracy = 0.0
    analytic1multi_accuracy = 0.0
    FD1multi_accuracy = 0.0
    FD1chunk1multi_accuracy = 0.0
    julia1_accuracy = 0.0
    FD1_accuracy = 0.0
    FD1chunk_accuracy = 0.0
    FD2hes_accuracy = 0.0
    FD2jac_accuracy = 0.0
    FD2jac1chunk_accuracy = 0.0
    tspan = (0.0,time)
    
    FD_first = FD(Order1(), ODE, x0, params, tspan)
    FD1chunk = FD(Order1(), ODE, x0, params, tspan, chunksize = 1)
    FD1chunk1multi = FD(Order1(), ODE, x0, params, tspan, chunksize = 1, multi = true)
    FD1chunk1multi_prediction = predict(Order1(), FD1chunk1multi, perturb)
    FD_first_prediction = predict(Order1(), FD_first, perturb)
    FD1chunk_prediction = predict(Order1(), FD1chunk, perturb)
    analytic_first = analytic_method(Order1(), ODE, x0, params, tspan, epsilon1; kwargs...)
    analytic_first_prediction = predict(Order1(), analytic_first, perturb)
    analyticfirstmulti = analytic_method_multi(Order1(), ODE, x0, params, tspan, epsilon1; kwargs...)
    analyticfirstmulti_prediction = predict(Order1(), analyticfirstmulti, perturb)
    FD1multi = FD(Order1(), ODE, x0, params, tspan, multi = true)
    FD1multi_prediction = predict(Order1(), FD1multi, perturb)
    exact = predictExact(ODE, x0, params, tspan, perturb; kwargs...)
    julia_first = DES(Order1(), ODE, x0, params, tspan; kwargs...)
    julia_first_prediction = predict(Order1(), julia_first, perturb)

    FDhes_second = FD(Order2(), ODE, x0, params, tspan)
    FDhes_second_prediction = predict(Order2(), FD_first, FDhes_second, perturb)
    FDjac_second = FD2(Order2(), ODE, x0, params, tspan)
    FDjac_second_prediction = predict(Order2(), FD_first, FDjac_second, perturb)
    FD2jac1chunk = FD2(Order2(), ODE, x0, params, tspan, chunksize = 1)
    FD2jac1chunk_prediction = predict(Order2(), FD1chunk, FD2jac1chunk, perturb)
    FD2multi = FD2(Order2(), ODE, x0, params, tspan, multi = true)
    FD2multi_prediction = predict(Order2(), FD1multi, FD2multi, perturb)
    FD2chunk1multi = FD2(Order2(), ODE, x0, params, tspan, chunksize = 1, multi = true)
    FD2chunk1multi_prediction = predict(Order2(), FD1chunk1multi, FD2chunk1multi, perturb)
    if !excludeDES2
        julia_second = DES(Order2(), ODE, x0, params, tspan, sensealg; kwargs...)
        julia_second_prediction = predict(Order2(), julia_first, julia_second, perturb)
    end
    if !excludeDES2_nonadjoint
        nonadjoint_second = DES(Order2(), ODE, x0, params, tspan, ForwardDiffOverAdjoint(ForwardSensitivity(autodiff=false)))
        nonadjoint_second_prediction = predict(Order2(), julia_first, nonadjoint_second, perturb)
    end
    
    for param = 1:length(params)
        analytic_second = analytic_method(Order2(), ODE, x0, params, tspan, epsilon2[param]; kwargs...)
        analytic_second_prediction = predict(Order2(), analytic_first, analytic_second, perturb)
        analyticsecondmulti = analytic_method_multi(Order2(), ODE, x0, params, tspan, epsilon2[param]; kwargs...)
        analyticsecondmulti_prediction = predict(Order2(), analytic_first, analyticsecondmulti, perturb)
        for compartment = 1:length(x0)
            #compute distance between predicted and actual trajectories
            actual = exact[param][compartment]
            FD1_accuracy += euclidean_dist(actual, FD_first_prediction[compartment][:, 1+param])
            FD1chunk_accuracy += euclidean_dist(actual, FD1chunk_prediction[compartment][:, 1+param])
            julia1_accuracy += euclidean_dist(actual, julia_first_prediction[compartment][:, 1+param])
            analytic1_accuracy += euclidean_dist(actual, analytic_first_prediction[compartment][:, 1+param])
            analytic1multi_accuracy += euclidean_dist(actual, analyticfirstmulti_prediction[compartment][:, 1+param])
            FD1multi_accuracy += euclidean_dist(actual, FD1multi_prediction[compartment][:, 1+param])
            FD1chunk1multi_accuracy += euclidean_dist(actual, FD1chunk1multi_prediction[compartment][:, 1+param])
            analytic2_accuracy += euclidean_dist(actual, analytic_second_prediction[compartment][:, 1+param])
            analytic2multi_accuracy += euclidean_dist(actual, analyticsecondmulti_prediction[compartment][:, 1+param])
            FD2hes_accuracy += euclidean_dist(actual, FDhes_second_prediction[compartment][:, 1+param])
            FD2jac_accuracy += euclidean_dist(actual, FDjac_second_prediction[compartment][:, 1+param])
            FD2jac1chunk_accuracy += euclidean_dist(actual, FD2jac1chunk_prediction[compartment][:, 1+param])
            FD2multi_accuracy += euclidean_dist(actual, FD2multi_prediction[compartment][:, 1+param])
            FD2chunk1multi_accuracy += euclidean_dist(actual, FD2chunk1multi_prediction[compartment][:, 1+param])
            if !excludeDES2
                julia2_accuracy += euclidean_dist(actual, julia_second_prediction[compartment][:, 1+param])
            end
            if !excludeDES2_nonadjoint
                nonadjoint2_accuracy += euclidean_dist(actual, nonadjoint_second_prediction[compartment][:, 1+param])
            end
        end
    end
    
    all_methods = [FD1_accuracy, FD1chunk_accuracy, julia1_accuracy, analytic1_accuracy, analytic1multi_accuracy, FD1multi_accuracy, FD1chunk1multi_accuracy, FD2hes_accuracy, FD2jac_accuracy, FD2jac1chunk_accuracy, julia2_accuracy, nonadjoint2_accuracy, analytic2_accuracy, analytic2multi_accuracy, FD2multi_accuracy, FD2chunk1multi_accuracy]
    all_methods ./= (length(x0) * length(params))
    return all_methods
end

function benchmarkSIR(t, N, n, change, sensealg, epsilon1, excludeDES2=false, excludeDES2_nonadjoint=false; kwargs...)
    x0 = [N - n, n, 0.0]
    params = [0.105, 0.12, N]
    tspan = (0.0,t)
    ODE = DiffEqBase.ODEFunction(SIR)
    times = benchmarkSIRTime(x0, params, tspan, epsilon1, ODE, sensealg, excludeDES2, excludeDES2_nonadjoint; kwargs...)
    return times
end

function benchmarkCARRGO(t, change, sensealg, epsilon1, excludeDES2=false, excludeDES2_nonadjoint=false; kwargs...)
    x0 = [1.25e4, 6.25e2] 
    params = [6.0e-9, 3.0e-11, 1.0e-6, 6.0e-2, 1.0e9]
    tspan = (0.0,t)
    ODE = DiffEqBase.ODEFunction(CARRGO)

    times = benchmarkSIRTime(x0, params, tspan, epsilon1, ODE, sensealg, excludeDES2, excludeDES2_nonadjoint; kwargs...)

    return times
end

function benchmarkMCC(t, change, sensealg, epsilon1, excludeDES2=false, excludeDES2_nonadjoint=false; kwargs...)
    x0 = [0.1, 0.05, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    params = [0.05, 0.025, 0.4, 0.1, 0.175, 1.5, 0.2, 0.1, 1.0, 0.1, 0.005, 0.1, 2.0, 0.4, 0.15]
    tspan = (0.0,t)
    ODE = DiffEqBase.ODEFunction(MCC)

    times = benchmarkSIRTime(x0, params, tspan, epsilon1, ODE, sensealg, excludeDES2, excludeDES2_nonadjoint; kwargs...)

    return times
end


function benchmarkSIRAccuracy(x0, params, tspan, epsilon1, epsilon2, ODE, del_param; kwargs...)
    accuracy = zeros(6)
    perturb = del_param * params
    #calculate derivatives
    analytic_first = analytic_method(Order1(), ODE, x0, params, tspan, epsilon1; kwargs...)
    analytic_second = analytic_method(Order2(), ODE, x0, params, tspan, epsilon2; kwargs...)
    julia_first = DES(Order1(), ODE, x0, params, tspan; kwargs...)
    julia_second = DES(Order2(), ODE, x0, params, tspan; kwargs...)
    #calculate trajectories
    original = predictExact(ODE, x0, params, tspan, zeros(length(params)); kwargs...)
    exact = predictExact(ODE, x0, params, tspan, perturb; kwargs...)
    analytic_first_prediction = predict(Order1(), analytic_first, perturb)
    analytic_second_prediction = predict(Order2(), analytic_first, analytic_second, perturb)
    julia_first_prediction = predict(Order1(), julia_first, perturb)
    julia_second_prediction = predict(Order2(), julia_first, julia_second, perturb)
    #calculate accuracies
    for compartment = 1:length(x0)
        for param = 1:length(params)
            #compute distance between predicted and actual trajectories
            actual = exact[param][compartment]
            accuracy[1] += euclidean_dist(actual, julia_first_prediction[compartment][:, 1+param])
            accuracy[2] += euclidean_dist(actual, analytic_first_prediction[compartment][:, 1+param])
            accuracy[4] += euclidean_dist(actual, julia_second_prediction[compartment][:, 1+param])
            accuracy[5] += euclidean_dist(actual, analytic_second_prediction[compartment][:, 1+param])
        end
    end
    #accuracy of mutithreading is the same as the accuracy of the non-multi-threaded version (don't need to rebenchmark; save time)
    accuracy[3] = accuracy[2] 
    accuracy[6] = accuracy[5]
    accuracy ./= (length(x0)*length(params)) #average over all compartments and parameters
    return accuracy
end

function benchmarkSIRTime(x0, params, tspan, epsilon1, ODE, sensealg, excludeDES2, excludeDES2_nonadjoint; kwargs...)
    times = zeros(16)
    epsilon2 = sqrt(epsilon1)
    sense = ForwardDiffOverAdjoint(ForwardSensitivity(autodiff=false))
    times[1] = median(@benchmark FD(Order1(), $ODE, $x0, $params, $tspan; $kwargs)).time
    times[2] = median(@benchmark FD(Order1(), $ODE, $x0, $params, $tspan; chunksize = 1, $kwargs)).time
    times[3] = median(@benchmark DES(Order1(), $ODE, $x0, $params, $tspan; $kwargs)).time
    times[4] = median(@benchmark analytic_method(Order1(), $ODE, $x0, $params, $tspan, $epsilon1; $kwargs)).time
    times[5] = median(@benchmark analytic_method_multi(Order1(), $ODE, $x0, $params, $tspan, $epsilon1; $kwargs)).time
    times[6] = median(@benchmark FD(Order1(), $ODE, $x0, $params, $tspan; multi = true, $kwargs)).time
    times[7] = median(@benchmark FD(Order1(), $ODE, $x0, $params, $tspan; chunksize = 1, multi = true, $kwargs)).time
    times[8] = median(@benchmark FD(Order2(), $ODE, $x0, $params, $tspan; $kwargs)).time
    times[9] = median(@benchmark FD2(Order2(), $ODE, $x0, $params, $tspan; $kwargs)).time
    times[10] = median(@benchmark FD2(Order2(), $ODE, $x0, $params, $tspan; chunksize = 1, $kwargs)).time
    if !excludeDES2
        times[11] = median(@benchmark DES(Order2(), $ODE, $x0, $params, $tspan, $sensealg; $kwargs)).time
    end
    if !excludeDES2_nonadjoint
        times[12] = median(@benchmark DES(Order2(), $ODE, $x0, $params, $tspan, $sense)).time
    end
    times[13] = median(@benchmark analytic_method(Order2(), $ODE, $x0, $params, $tspan, $epsilon2; $kwargs)).time
    times[14] = median(@benchmark analytic_method_multi(Order2(), $ODE, $x0, $params, $tspan, $epsilon2; $kwargs)).time
    times[15] = median(@benchmark FD2(Order2(), $ODE, $x0, $params, $tspan; multi = true, $kwargs)).time
    times[16] = median(@benchmark FD2(Order2(), $ODE, $x0, $params, $tspan; chunksize = 1, multi = true, $kwargs)).time
    # convert ns to μs
    
    times ./= 1e3
    return times
end
