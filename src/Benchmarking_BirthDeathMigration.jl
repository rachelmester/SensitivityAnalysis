#=
Code for Birth Notes on Birth-Death Migration Processes (Kenneth Lange)
code by Rachel Mester
last updated October 21 2020
=#

using BenchmarkTools, DelimitedFiles

#=
Realistic Parameter Generation for Branching Process
=#

"""
    function createRealisticSubCriticalParams(n::Integer)::Array

This function creates set of realistic parameters β (nx1), δ (nx1), and
λ (nxn) for a branching process that is guaranteed to be subcritical.  In
the returned parameter array, column 1 is β, column 2 is δ, and columns
3 to end are the full migration matrix (where rows sum to zero).

# Example
```julia-repl
julia> createRealisticSubCriticalParams(2)
2×4 Array{Float64,2}:
 0.0757347  0.105735   -0.000373983   0.000373983
 0.0514148  0.0814148   0.000335731  -0.000335731
 ```
"""
function createRealisticSubCriticalParams(n::Integer)::Array
    birth = vec(rand(n,1)) .* .11 .+ .05
    death = birth .+ .03
    migration = rand(n,n) .* 0.0003 .+ .00016
    migration = migration_generator(migration)
    return [birth death migration]
end

"""
    function createRealisticParams(n::Integer)::Array

This function creates set of realistic parameters β (nx1), δ (nx1), and
λ (nxn) for a branching process.  In the returned parameter array, column
1 is β, column 2 is δ, and columns 3 to end are the full migration
matrix (where rows sum to zero).

# Example
```julia-repl
julia> createRealisticParams(2)
2×4 Array{Float64,2}:
 0.0661948  0.122163  -0.000270934   0.000270934
 0.075763   0.167761   0.000281205  -0.000281205
 ```
"""
function createRealisticParams(n::Integer)::Array
    birth=vec(rand(n,1)) .* .11 .+ .05
    death=vec(rand(n,1)) .* .14 .+ .05
    migration=rand(n,n) .* .0003 .+ .00016
    migration=migration_generator(migration)
    return [birth death migration]
end

"""
    function exportBranchingParams()

This function exports ten sets each of realistic parameters β (nx1),
δ (nx1), and λ (nxn) for branching processes of size
n=10,100,1000 and with nonspecified, subcritical, symmetric
characteristics.  These parameters are exported to "<n>RealisticParams.csv",
"<n>RealisticSubCriticalParams.csv", "<n>RealisticSymmetricParams.csv".

In the exported documents, each block of n rows constitutes a parameter matrix,
where column 1 is the birth parameters, column 2 is the death parameters, and
columns 3 to end are the full migration matrix (with rows that sum to zero).
"""
function exportBranchingParams()
    N=[10,100,1000]
    t=10
    for n in N
        allParams=zeros(n*t,n+2)
        for i=1:t
            allParams[n*(i-1)+1:n*i,:]=createRealisticParams(n)
        end
        writedlm(string(n,"RealisticParams.csv"),allParams,',')
        allParams=zeros(n*t,n+2)
        for i=1:t
            allParams[n*(i-1)+1:n*i,:]=createRealisticSubCriticalParams(n)
        end
        writedlm(string(n,"RealisticSubCriticalParams.csv"),allParams,',')
    end
end

#=
Realistic Parameter Generation for the SIR Model
=#

"""
    function exportSIRParams()

This function exports ten sets of realistic parameters δ and η for the
SIR model to "RealisticSIRParams.csv".


In the exported documents, each column represents a set of parameters, with row
1 being the δ and row 2 being the η parameter.
"""
function exportSIRParams()
    trials=10
    writedlm("RealisticSIRParams.csv",transpose([.0417 .+ .0171*rand(trials) .0012 .+ .4788*rand(trials)]),',')
end

#=
Benchmarking methods for Branching Processes
=#

"""
    function benchmarkBranching_bySizeDelta()

This function calculates time, memory, and loss of precision for the
branching process methods over n=10,100,1000 averaged over the
ten sets of parameters generated for each size, and exports the results to
"BranchingbySizeDelta.csv".  The derivatives are calculated with θ=δ[1].
Threshold=1e-20 and step size 1e-6 are used where applicable.
"""
function benchmarkBranching_bySizeDelta()
    trials=10
    N=[10,100,1000]
    Fsub=[cumulative_inverse,cumulative_iterative,dAddelta_inverse,dAddelta_iterative,dAddelta_complex]
    Freg=[extinction_probability,deddelta_inverse,de_ddelta,deddelta_complex,drho_dtheta,dmean_ddelta]
    iter=0
    vals=zeros(length(N)*(length(Fsub)+length(Freg)),5)
    for n in N
        subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
        for f in Fsub
            iter=iter+1
            vals[iter,2]=n*trials
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration))
                diff=sum(abs.(f(birth,death,migration).-f(birth_s,death_s,migration_s)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
        regParams=readdlm(string(n,"RealisticParams.csv"),',')
        for f in Freg
            iter=iter+1
            vals[iter,2]=n*trials
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration))
                diff=sum(abs.(f(birth,death,migration).-f(birth_s,death_s,migration_s)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingbySizeDelta.csv",vals,',')
    return vals
end

"""
    function benchmarkBranching_bySizeBeta()

This function calculates time, memory, and loss of precision for the
branching process methods over n=10,100,1000 averaged over the ten sets of
parameters generated for each size, and exports the results to
"BranchingbySizeBeta.csv".  The derivatives are calculated with θ=β[1].
Threshold=1e-20 and step size 1e-6 are used where applicable.
"""
function benchmarkBranching_bySizeBeta()
    #trials=10
    trials = 1
    N=[10, 100, 1000]
    #N = [10]
    Fsub = [dAdbeta_inverse, dAdbeta_complexinverse]
    #Fsub=[dAdbeta_inverse,dAdbeta_iterative,dAdbeta_complex]
    #Freg=[dedbeta_inverse,de_dbeta,dedbeta_complex,drho_dbeta,dmean_dbeta]
    Freg=[de_dbeta,dedbeta_complex]
    iter=0
    vals=zeros(length(N)*(length(Fsub)+length(Freg)),5)
    for n in N
        subParams=readdlm(string("../data/",n,"RealisticSubCriticalParams.csv"),',')
        for f in Fsub
            iter=iter+1
            vals[iter,2]=n*trials
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                #birth_s=convert(Array{Float32},birth)
                #death_s=convert(Array{Float32},death)
                #migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration, 1e-18, 1e-12))
                #diff=sum(abs.(f(birth,death,migration).-f(birth_s,death_s,migration_s)))/n
                vals[iter,3]=vals[iter,3]+test.time
                #vals[iter,4]=vals[iter,4]+test.memory
                #vals[iter,5]=vals[iter,5]+diff
            end
            #println(vals[iter,:]./trials)
        end
        regParams=readdlm(string("../data/",n,"RealisticParams.csv"),',')
        for f in Freg
            iter=iter+1
            vals[iter,2]=n*trials
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                #birth_s=convert(Array{Float32},birth)
                #death_s=convert(Array{Float32},death)
                #migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration, 1e-18, 1e-12))
                #diff=sum(abs.(f(birth,death,migration).-f(birth_s,death_s,migration_s)))/n
                vals[iter,3]=vals[iter,3]+test.time
                #vals[iter,4]=vals[iter,4]+test.memory
                #vals[iter,5]=vals[iter,5]+diff
            end
            #println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    #writedlm("BranchingbySizeBeta.csv",vals,',')
    return vals
end

"""
    function benchmarkBranching_bySizeLambda()

This function calculates time, memory, and loss of precision for the
branching process methods over n=10,100,1000 averaged over the ten sets of
parameters generated for each size, and exports the results to
"BranchingbySizeLambda.csv".  The derivatives are calculated with
θ=λ[1,2].  Threshold=1e-20 and step size 1e-6 are used where
applicable.
"""
function benchmarkBranching_bySizeLambda()
    trials=10
    N=[10,100,1000]
    Fsub=[dAdlambda_inverse,dAdlambda_iterative,dAdlambda_complex]
    Freg=[dedlambda_inverse,de_dlambda,dedlambda_complex,drho_dlambda,dmean_dlambda]
    iter=0
    vals=zeros(length(N)*(length(Fsub)+length(Freg)),5)
    for n in N
        subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
        for f in Fsub
            iter=iter+1
            vals[iter,2]=n*trials
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration))
                diff=sum(abs.(f(birth,death,migration).-f(birth_s,death_s,migration_s)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
        regParams=readdlm(string(n,"RealisticParams.csv"),',')
        for f in Freg
            iter=iter+1
            vals[iter,2]=n*trials
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration))
                diff=sum(abs.(f(birth,death,migration).-f(birth_s,death_s,migration_s)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingbySizeLambda.csv",vals,',')
    return vals
end

"""
    function benchmarkBranching_byThresholdDelta()

This function calculates time, memory, and loss of precision for the
branching process methods over
threshold=1e-8,1e-12,1e-16, averaged over the ten sets
of parameters generated for n=10, and exports the results to
"BranchingbyThresholdDelta.csv".  The derivatives are calculated with θ=δ[1].
Step size 1e-6 is used where applicable.
"""
function benchmarkBranching_byThresholdDelta()
    trials=10
    n=10
    threshold=[1e-8,1e-12,1e-16]
    Fsub=[cumulative_iterative,dAddelta_iterative,dAddelta_complex]
    Freg=[extinction_probability,de_ddelta,deddelta_complex,dmean_ddelta]
    iter=0
    vals=zeros(length(threshold)*(length(Fsub)+length(Freg)),5)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for thresh in threshold
        for f in Fsub
            iter=iter+1
            vals[iter,2]=thresh*trials
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh))
                diff=sum(abs.(f(birth,death,migration,thresh).-f(birth_s,death_s,migration_s,thresh)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
        for f in Freg
            iter=iter+1
            vals[iter,2]=thresh*trials
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh))
                diff=sum(abs.(f(birth,death,migration,thresh).-f(birth_s,death_s,migration_s,thresh)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingbyThresholdDelta.csv",vals,',')
    return vals
end

"""
    function benchmarkBranching_byThresholdBeta()

This function calculates time, memory, and loss of precision for the
branching process methods over threshold=1e-8,1e-12,1e-16 averaged over the ten
sets of parameters generated for n=10, and exports the results to
"BranchingbyThresholdBeta.csv".  The derivatives are calculated with
θ=β[1].  Step size 1e-6 is used where applicable.
"""
function benchmarkBranching_byThresholdBeta()
    trials=10
    n=10
    threshold=[1e-8,1e-12,1e-16]
    Fsub=[dAdbeta_iterative,dAdbeta_complex]
    Freg=[de_dbeta,dedbeta_complex,dmean_dbeta]
    iter=0
    vals=zeros(length(threshold)*(length(Fsub)+length(Freg)),5)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for thresh in threshold
        for f in Fsub
            iter=iter+1
            vals[iter,2]=thresh*trials
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh))
                diff=sum(abs.(f(birth,death,migration,thresh).-f(birth_s,death_s,migration_s,thresh)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
        for f in Freg
            iter=iter+1
            vals[iter,2]=thresh*trials
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh))
                diff=sum(abs.(f(birth,death,migration,thresh).-f(birth_s,death_s,migration_s,thresh)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingbyThresholdBeta.csv",vals,',')
    return vals
end

"""
    function benchmarkBranching_byThresholdLambda()

This function calculates time, memory, and loss of precision for the
branching process methods over threshold=1e-8,1e-12,1e-16 averaged over the ten
sets of parameters generated for n=10, and exports the results to
"BranchingbyThresholdLambda.csv".  The derivatives are calculated with
θ=λ[1,2].  Step size 1e-6 is used where applicable.
"""
function benchmarkBranching_byThresholdLambda()
    trials=10
    n=10
    threshold=[1e-8,1e-12,1e-16]
    Fsub=[dAdlambda_iterative,dAdlambda_complex]
    Freg=[de_dlambda,dedlambda_complex,dmean_dlambda]
    iter=0
    vals=zeros(length(threshold)*(length(Fsub)+length(Freg)),5)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for thresh in threshold
        for f in Fsub
            iter=iter+1
            vals[iter,2]=thresh*trials
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh))
                diff=sum(abs.(f(birth,death,migration,thresh).-f(birth_s,death_s,migration_s,thresh)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
        for f in Freg
            iter=iter+1
            vals[iter,2]=thresh*trials
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh))
                diff=sum(abs.(f(birth,death,migration,thresh).-f(birth_s,death_s,migration_s,thresh)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingbyThresholdLambda.csv",vals,',')
    return vals
end

"""
    function benchmarkBranching_byStepDelta()

This function calculates time, memory, and loss of precision for the
branching process methods over step=1e-3,1e-6,1e-9 averaged
over the ten sets of parameters generated for n=10, and exports the results to
"BranchingbyStepDelta.csv".  The derivatives are calculated with θ=δ[1].
Threshold 1e-20 is used where applicable.
"""
function benchmarkBranching_byStepDelta()
    trials=10
    n=10
    thresh=1e-20
    eps=[1e-3,1e-6,1e-9]
    Fsub=[dAddelta_complex]
    Freg=[deddelta_complex]
    iter=0
    vals=zeros(length(eps)*(length(Fsub)+length(Freg)),5)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for step in eps
        for f in Fsub
            iter=iter+1
            vals[iter,2]=step*trials
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh,$step))
                diff=sum(abs.(f(birth,death,migration,thresh,step).-f(birth_s,death_s,migration_s,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
        for f in Freg
            iter=iter+1
            vals[iter,2]=step*trials
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh,$step))
                diff=sum(abs.(f(birth,death,migration,thresh,step).-f(birth_s,death_s,migration_s,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingbyStepDelta.csv",vals,',')
    return vals
end

"""
    function benchmarkBranching_byStepBeta()

This function calculates time, memory, and loss of precision for the
branching process methods over step=1e-3,1e-6,1e-9 averaged over the ten sets
of parameters generated for n=10, and exports the results to
"BranchingbyStepBeta.csv".  The derivatives are calculated with θ=β[1].
Threshold 1e-20 is used where applicable.
"""
function benchmarkBranching_byStepBeta()
    trials=10
    n=10
    thresh=1e-20
    eps=[1e-3,1e-6,1e-9]
    Fsub=[dAdbeta_complex]
    Freg=[dedbeta_complex]
    iter=0
    vals=zeros(length(eps)*(length(Fsub)+length(Freg)),5)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for step in eps
        for f in Fsub
            iter=iter+1
            vals[iter,2]=step*trials
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh,$step))
                diff=sum(abs.(f(birth,death,migration,thresh,step).-f(birth_s,death_s,migration_s,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
        for f in Freg
            iter=iter+1
            vals[iter,2]=step*trials
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh,$step))
                diff=sum(abs.(f(birth,death,migration,thresh,step).-f(birth_s,death_s,migration_s,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingbyStepBeta.csv",vals,',')
    return vals
end

"""
    function benchmarkBranching_byStepLambda()

This function calculates time, memory, and loss of precision for the
branching process methods over step=1e-3,1e-6,1e-9 averaged over the ten sets
of parameters generated for n=10, and exports the results to
"BranchingbyStepLambda.csv".  The derivatives are calculated with
θ=λ[1,2].  Threshold 1e-20 is used where applicable.
"""
function benchmarkBranching_byStepLambda()
    trials=10
    n=10
    thresh=1e-20
    eps=[1e-3,1e-6,1e-9]
    Fsub=[dAdlambda_complex]
    Freg=[dedlambda_complex]
    iter=0
    vals=zeros(length(eps)*(length(Fsub)+length(Freg)),5)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for step in eps
        for f in Fsub
            iter=iter+1
            vals[iter,2]=step*trials
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh,$step))
                diff=sum(abs.(f(birth,death,migration,thresh,step).-f(birth_s,death_s,migration_s,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
        for f in Freg
            iter=iter+1
            vals[iter,2]=step*trials
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                birth_s=convert(Array{Float32},birth)
                death_s=convert(Array{Float32},death)
                migration_s=convert(Array{Float32,2},migration)
                test=median(@benchmark $f($birth,$death,$migration,$thresh,$step))
                diff=sum(abs.(f(birth,death,migration,thresh,step).-f(birth_s,death_s,migration_s,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingbyStepLambda.csv",vals,',')
    return vals
end

"""
    function compareBranching_bySizeDelta()

This function calculates loss of accuracy for the branching process methods,
comparing outomes of inverse, iterative, and complex methods for
N=10,100,1000 averaged over the ten sets of parameters generated,
and exports the results to "BranchingComparebySizeDelta.csv".  The derivatives are
calculated with θ=δ[1].  Threshold 1e-20 and step size 1e-6 are used
where applicable.
"""
function compareBranching_bySizeDelta()
    trials=10
    N=[10,100,1000]
    F1_sub=[cumulative_inverse,dAddelta_inverse,dAddelta_inverse,dAddelta_iterative]
    F2_sub=[cumulative_iterative,dAddelta_iterative,dAddelta_complex,dAddelta_complex]
    F1_reg=[de_ddelta,de_ddelta,deddelta_complex]
    F2_reg=[deddelta_complex,deddelta_inverse,deddelta_inverse]
    num_sub=length(F1_sub)
    num_reg=length(F1_reg)
    iter=0
    vals=zeros(length(N)*(num_sub+num_reg),3)
    for n in N
        subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
        for a=1:num_sub
            iter=iter+1
            vals[iter,2]=n*trials
            f1=F1_sub[a]
            f2=F2_sub[a]
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration).-f2(birth,death,migration)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
        regParams=readdlm(string(n,"RealisticParams.csv"),',')
        for a=1:num_reg
            iter=iter+1
            vals[iter,2]=n*trials
            f1=F1_reg[a]
            f2=F2_reg[a]
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration).-f2(birth,death,migration)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingComparebySizeDelta.csv",vals,',')
    return vals
end

"""
    function compareBranching_bySizeBeta()

This function calculates loss of accuracy for the branching process methods,
comparing outomes of inverse, iterative, and complex methods for
N=10,100,1000 averaged over the ten sets of parameters generated, and exports
the results to "BranchingComparebySizeBeta.csv".  The derivatives are
calculated with θ=β[1].  Threshold 1e-20 and step size 1e-6 are used
where applicable.
"""
function compareBranching_bySizeBeta()
    trials=10
    N=[10,100,1000]
    F1_sub=[dAdbeta_inverse,dAdbeta_inverse,dAdbeta_iterative]
    F2_sub=[dAdbeta_iterative,dAdbeta_complex,dAdbeta_complex]
    F1_reg=[de_dbeta,de_dbeta,dedbeta_complex]
    F2_reg=[dedbeta_complex,dedbeta_inverse,dedbeta_inverse]
    num_sub=length(F1_sub)
    num_reg=length(F1_reg)
    iter=0
    vals=zeros(length(N)*(num_sub+num_reg),3)
    for n in N
        subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
        for a=1:num_sub
            iter=iter+1
            vals[iter,2]=n*trials
            f1=F1_sub[a]
            f2=F2_sub[a]
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration).-f2(birth,death,migration)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
        regParams=readdlm(string(n,"RealisticParams.csv"),',')
        for a=1:num_reg
            iter=iter+1
            vals[iter,2]=n*trials
            f1=F1_reg[a]
            f2=F2_reg[a]
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration).-f2(birth,death,migration)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingComparebySizeBeta.csv",vals,',')
    return vals
end

"""
    function compareBranching_bySizeLambda()

This function calculates loss of accuracy for the branching process methods,
comparing outomes of inverse, iterative, and complex methods for
N=10,100,1000 averaged over the ten sets of parameters generated, and exports
the results to "BranchingComparebySizeLambda.csv".  The derivatives are
calculated with θ=λ[1,2].  Threshold 1e-20 and step size 1e-6 are
used where applicable.
"""
function compareBranching_bySizeLambda()
    trials=10
    N=[10,100,1000]
    F1_sub=[dAdlambda_inverse,dAdlambda_inverse,dAdlambda_iterative]
    F2_sub=[dAdlambda_iterative,dAdlambda_complex,dAdlambda_complex]
    F1_reg=[de_dlambda,de_dlambda,dedlambda_complex]
    F2_reg=[dedlambda_complex,dedlambda_inversse,dedlambda_inverse]
    num_sub=length(F1_sub)
    num_reg=length(F1_reg)
    iter=0
    vals=zeros(length(N)*(num_sub+num_reg),3)
    for n in N
        subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
        for a=1:num_sub
            iter=iter+1
            vals[iter,2]=n*trials
            f1=F1_sub[a]
            f2=F2_sub[a]
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration).-f2(birth,death,migration)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
        regParams=readdlm(string(n,"RealisticParams.csv"),',')
        for a=1:num_reg
            iter=iter+1
            vals[iter,2]=n*trials
            f1=F1_reg[a]
            f2=F2_reg[a]
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration).-f2(birth,death,migration)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingComparebySizeLambda.csv",vals,',')
    return vals
end

"""
    function compareBranching_byThresholdDelta()

This function calculates loss of accuracy for the branching process methods,
comparing outomes of inverse, iterative, and complex methods for
threshold=1e-8,1e-12,1e-16 averaged over the ten sets
of parameters generated for n=10, and exports the results to
"BranchingComparebyThresholdDelta.csv".  The derivatives are calculated with
θ=δ[1].  Step size 1e-6 is used where applicable.
"""
function compareBranching_byThresholdDelta()
    trials=10
    n=10
    thresholds=[1e-8,1e-12,1e-16]
    F1_sub=[cumulative_inverse,dAddelta_inverse,dAddelta_inverse,dAddelta_iterative]
    F2_sub=[cumulative_iterative,dAddelta_iterative,dAddelta_complex,dAddelta_complex]
    F1_reg=[de_ddelta,de_ddelta,deddelta_complex]
    F2_reg=[deddelta_complex,deddelta_inverse,deddelta_inverse]
    num_sub=length(F1_sub)
    num_reg=length(F1_reg)
    iter=0
    vals=zeros(length(thresholds)*(num_sub+num_reg),3)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for thresh in thresholds
        for a=1:num_sub
            iter=iter+1
            vals[iter,2]=thresh*trials
            f1=F1_sub[a]
            f2=F2_sub[a]
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh).-f2(birth,death,migration,thresh)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
        for a=1:num_reg
            iter=iter+1
            vals[iter,2]=thresh*trials
            f1=F1_reg[a]
            f2=F2_reg[a]
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh).-f2(birth,death,migration,thresh)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingComparebyThresholdDelta.csv",vals,',')
    return vals
end

"""
    function compareBranching_byThresholdBeta()

This function calculates loss of accuracy for the branching process methods,
comparing outomes of inverse, iterative, and complex methods for
threshold=1e-8,1e-12,1e-16 averaged over the ten sets of parameters generated
for n=10, and exports the results to "BranchingComparebyThresholdBeta.csv".
The derivatives are calculated with θ=β[1].  Step size 1e-6 is used where
applicable.
"""
function compareBranching_byThresholdBeta()
    trials=10
    n=10
    thresholds=[1e-8,1e-12,1e-16]
    F1_sub=[dAdbeta_inverse,dAdbeta_inverse,dAdbeta_iterative]
    F2_sub=[dAdbeta_iterative,dAdbeta_complex,dAdbeta_complex]
    F1_reg=[de_dbeta,de_dbeta,dedbeta_complex]
    F2_reg=[dedbeta_complex,dedbeta_inverse,dedbeta_inverse]
    num_sub=length(F1_sub)
    num_reg=length(F1_reg)
    iter=0
    vals=zeros(length(thresholds)*(num_sub+num_reg),3)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for thresh in thresholds
        for a=1:num_sub
            iter=iter+1
            vals[iter,2]=thresh*trials
            f1=F1_sub[a]
            f2=F2_sub[a]
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh).-f2(birth,death,migration,thresh)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
        for a=1:num_reg
            iter=iter+1
            vals[iter,2]=thresh*trials
            f1=F1_reg[a]
            f2=F2_reg[a]
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh).-f2(birth,death,migration,thresh)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingComparebyThresholdBeta.csv",vals,',')
    return vals
end

"""
    function compareBranching_byThresholdLambda()

This function calculates loss of accuracy for the branching process methods,
comparing outomes of inverse, iterative, and complex methods for
threshold=1e-8,1e-12,1e-16 averaged over the ten sets of parameters generated
for n=10, and exports the results to "BranchingComparebyThresholdLambda.csv".
The derivatives are calculated with θ=λ[1,2].  Step size 1e-6 is used
where applicable.
"""
function compareBranching_byThresholdLambda()
    trials=10
    n=10
    thresholds=[1e-8,1e-12,1e-16]
    F1_sub=[dAdlambda_inverse,dAdlambda_inverse,dAdlambda_iterative]
    F2_sub=[dAdlambda_iterative,dAdlambda_complex,dAdlambda_complex]
    F1_reg=[de_dlambda,de_dlambda,dedlambda_complex]
    F2_reg=[dedlambda_complex,dedlambda_inverse,dedlambda_inverse]
    num_sub=length(F1_sub)
    num_reg=length(F1_reg)
    iter=0
    vals=zeros(length(thresholds)*(num_sub+num_reg),3)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for thresh in thresholds
        for a=1:num_sub
            iter=iter+1
            vals[iter,2]=thresh*trials
            f1=F1_sub[a]
            f2=F2_sub[a]
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh).-f2(birth,death,migration,thresh)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
        for a=1:num_reg
            iter=iter+1
            vals[iter,2]=thresh*trials
            f1=F1_reg[a]
            f2=F2_reg[a]
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh).-f2(birth,death,migration,thresh)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingComparebyThresholdLambda.csv",vals,',')
    return vals
end

"""
    function compareBranching_byStepDelta

This function calculates loss of accuracy for the branching process methods,
comparing outomes of inverse, iterative, and complex methods for
step=1e-3,1e-6,1e-9 averaged over the ten sets of
parameters generated for n=10, and exports the results to
"BranchingComparebyStepDelta.csv".  The derivatives are calculated with
θ=δ[1].  Threshold 1e-20 is used where applicable.
"""
function compareBranching_byStepDelta()
    trials=10
    n=10
    thresh=1e-20
    eps=[1e-3,1e-6,1e-9]
    F1_sub=[dAddelta_inverse,dAddelta_iterative]
    F2_sub=[dAddelta_complex,dAddelta_complex]
    F1_reg=[de_ddelta,deddelta_inverse]
    F2_reg=[deddelta_complex,deddelta_complex]
    num_sub=length(F1_sub)
    num_reg=length(F1_reg)
    iter=0
    vals=zeros(length(eps)*(num_sub+num_reg),3)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for step in eps
        for a=1:num_sub
            iter=iter+1
            vals[iter,2]=step*trials
            f1=F1_sub[a]
            f2=F2_sub[a]
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh,step).-f2(birth,death,migration,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
        for a=1:num_reg
            iter=iter+1
            vals[iter,2]=step*trials
            f1=F1_reg[a]
            f2=F2_reg[a]
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh,step).-f2(birth,death,migration,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingComparebyStepDelta.csv",vals,',')
    return vals
end

"""
    function compareBranching_byStepBeta

This function calculates loss of accuracy for the branching process methods,
comparing outomes of inverse, iterative, and complex methods for
step=1e-3,1e-6,1e-9 averaged over the ten sets of parameters generated for n=10,
and exports the results to "BranchingComparebyStepBeta.csv".  The derivatives
are calculated with θ=β[1].  Threshold 1e-20 is used where applicable.
"""
function compareBranching_byStepBeta()
    trials=10
    n=10
    thresh=1e-20
    eps=[1e-3,1e-6,1e-9]
    F1_sub=[dAdbeta_inverse,dAdbeta_iterative]
    F2_sub=[dAdbeta_complex,dAdbeta_complex]
    F1_reg=[de_dbeta,dedbeta_inverse]
    F2_reg=[dedbeta_complex,dedbeta_complex]
    num_sub=length(F1_sub)
    num_reg=length(F1_reg)
    iter=0
    vals=zeros(length(eps)*(num_sub+num_reg),3)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for step in eps
        for a=1:num_sub
            iter=iter+1
            vals[iter,2]=step*trials
            f1=F1_sub[a]
            f2=F2_sub[a]
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh,step).-f2(birth,death,migration,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
        for a=1:num_reg
            iter=iter+1
            vals[iter,2]=step*trials
            f1=F1_reg[a]
            f2=F2_reg[a]
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh,step).-f2(birth,death,migration,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingComparebyStepBeta.csv",vals,',')
    return vals
end

"""
    function compareBranching_byStepLambda

This function calculates loss of accuracy for the branching process methods,
comparing outomes of inverse, iterative, and complex methods for
step=1e-3,1e-6,1e-9 averaged over the ten sets of parameters generated for n=10,
and exports the results to "BranchingComparebyStepLambda.csv".  The derivatives
are calculated with θ=λ[1,2].  Threshold 1e-20 is used where
applicable.
"""
function compareBranching_byStepLambda()
    trials=10
    n=10
    thresh=1e-20
    eps=[1e-3,1e-6,1e-9]
    F1_sub=[dAdlambda_inverse,dAdlambda_iterative]
    F2_sub=[dAdlambda_complex,dAdlambda_complex]
    F1_reg=[de_dlambda,dedlambda_inverse]
    F2_reg=[dedlambda_complex,dedlambda_complex]
    num_sub=length(F1_sub)
    num_reg=length(F1_reg)
    iter=0
    vals=zeros(length(eps)*(num_sub+num_reg),3)
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for step in eps
        for a=1:num_sub
            iter=iter+1
            vals[iter,2]=step*trials
            f1=F1_sub[a]
            f2=F2_sub[a]
            for r in 1:trials
                params=subParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh,step).-f2(birth,death,migration,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
        for a=1:num_reg
            iter=iter+1
            vals[iter,2]=step*trials
            f1=F1_reg[a]
            f2=F2_reg[a]
            for r in 1:trials
                params=regParams[n*(r-1)+1:n*r,:]
                birth=params[:,1]
                death=params[:,2]
                migration=params[:,3:end]
                diff=sum(abs.(f1(birth,death,migration,thresh,step).-f2(birth,death,migration,thresh,step)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("BranchingComparebyStepLambda.csv",vals,',')
    return vals
end

"""
    function branchingAccuracyByChangeDelta

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A, e, and ρ using the finite difference formulas
with the actual values of A, e, and ρ with the different value of δ[1].  This
method compares these values across dδ{.01*δ[1],.1*δ[1],1*δ[1]}
averaged over the ten sets of parameters generated, and exports the results to
"BranchingAccuracybyChangeDelta.csv". Threshold 1e-20, size n=10, and complex step
size 1e-6 are used.
"""
function branchingAccuracyByChangeDelta()
    n=10
    dels=[.01,.1,1]
    L=length(dels)
    vals=zeros(L,9)
    trials=10
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    for r in 1:trials
        params=subParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        A=cumulative_inverse(birth,death,migration)
        dA_1=dAddelta_inverse(birth,death,migration)
        dA_2=dAddelta_iterative(birth,death,migration)
        dA_3=dAddelta_complex(birth,death,migration)
        for d in 1:L
            change=dels[d]
            vals[d,1]=trials*change
            ddelta=death[1]*change
            A1_expected=A+(ddelta*dA_1)
            A2_expected=A+(ddelta*dA_2)
            A3_expected=A+(ddelta*dA_3)
            death[1]=death[1]+ddelta
            A_actual=cumulative_inverse(birth,death,migration)
            vals[d,2]=vals[d,2]+sum(abs.(A1_expected.-A_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(A2_expected.-A_actual))/n
            vals[d,4]=vals[d,4]+sum(abs.(A3_expected.-A_actual))/n
        end
    end
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for r in 1:trials
        params=regParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        e=extinction_probability(birth,death,migration)
        de_1=de_ddelta(birth,death,migration)
        de_2=deddelta_complex(birth,death,migration)
        de_3=deddelta_inverse(birth,death,migration)
        drho=drho_ddelta(birth,death,migration)
        rho=find_rho(calculate_generator(birth,death,migration),n)
        dmean=dmean_ddelta(birth,death,migration)
        mean=expected_number(calculate_generator(birth,death,migration),2)
        for d in 1:L
            change=dels[d]
            vals[d,1]=trials*change
            ddelta=death[1]*change
            rho_expected=rh+(ddelta*drho)
            e1_expected=e+(ddelta*de_1)
            e2_expected=e+(ddelta*de_2)
            e3_expected=e+(ddelta*de_3)
            mean_expected=mean+(ddelta*dmean)
            death[1]=death[1]+ddelta
            e_actual=extinction_probability(birth,death,migration)
            vals[d,5]=vals[d,5]+sum(abs.(e1_expected.-e_actual))/n
            vals[d,6]=vals[d,6]+sum(abs.(e2_expected.-e_actual))/n
            vals[d,8]=vals[d,8]+sum(abs.(e3_expected.-e_actual))/n
            rho_actual=find_rho(calculate_generator(birth,death,migration),n)
            vals[d,7]=vals[d,7]+abs(rho_actual-rho_expected)
            mean_actual=expected_number(calculate_generator(birth,death,migration),2)
            vals[d,9]=vals[d,9]+sum(abs.(mean_expected-mean_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("BranchingAccuracybyChangeDelta.csv",vals,',')
end

"""
    function branchingAccuracyByChangeBeta

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of β[1].  This
method compares these values across dβ={.01*β[1],.1*β[1],1*β[1]}
averaged over the ten sets of parameters generated, and exports the results to
"BranchingAccuracybyChangeBeta.csv". Threshold 1e-20, size n=10, and complex step
size 1e-6 are used.
"""
function branchingAccuracyByChangeBeta()
    n=10
    dels=[.01,.1,1]
    L=length(dels)
    vals=zeros(L,9)
    trials=10
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    for r in 1:trials
        params=subParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        A=cumulative_inverse(birth,death,migration)
        dA_1=dAdbeta_inverse(birth,death,migration)
        dA_2=dAdbeta_iterative(birth,death,migration)
        dA_3=dAdbeta_complex(birth,death,migration)
        for d in 1:L
            change=dels[d]
            vals[d,1]=trials*change
            dbeta=birth[1]*change
            A1_expected=A+(dbeta*dA_1)
            A2_expected=A+(dbeta*dA_2)
            A3_expected=A+(dbeta*dA_3)
            birth[1]=birth[1]+dbeta
            A_actual=cumulative_inverse(birth,death,migration)
            vals[d,2]=vals[d,2]+sum(abs.(A1_expected.-A_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(A2_expected.-A_actual))/n
            vals[d,4]=vals[d,4]+sum(abs.(A3_expected.-A_actual))/n
        end
    end
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for r in 1:trials
        params=regParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        e=extinction_probability(birth,death,migration)
        de_1=de_dbeta(birth,death,migration)
        de_2=dedbeta_complex(birth,death,migration)
        de_3=dedbeta_complex(birth,death,migration)
        drho=drho_dbeta(birth,death,migration)
        rho=find_rho(calculate_generator(birth,death,migration),n)
        dmean=dmean_ddelta(birth,death,migration)
        mean=expected_number(calculate_generator(birth,death,migration),2)
        for d in 1:L
            change=dels[d]
            vals[d,1]=trials*change
            dbeta=birth[1]*change
            e1_expected=e+(dbeta*de_1)
            e2_expected=e+(dbeta*de_2)
            e3_expected=e+(dbeta*de_3)
            mean_expected=mean+(dbeta*dmean)
            birth[1]=birth[1]+dbeta
            e_actual=extinction_probability(birth,death,migration)
            vals[d,5]=vals[d,5]+sum(abs.(e1_expected.-e_actual))/n
            vals[d,6]=vals[d,6]+sum(abs.(e2_expected.-e_actual))/n
            rho_actual=find_rho(calculate_generator(birth,death,migration),n)
            vals[d,7]=vals[d,7]+abs(rho_actual-rho_expected)
            vals[d,8]=vals[d,8]+sum(abs.(e3_expected.-e_actual))/n
            mean_actual=expected_number(calculate_generator(birth,death,migration),2)
            vals[d,9]=vals[d,9]+sum(abs.(mean_expected-mean_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("BranchingAccuracybyChangeBeta.csv",vals,',')
end

"""
    function branchingAccuracyByChangeLambda

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of λ[1,2].  This
method compares these values across
dλ={.01*λ[1,2],.1*λ[1,2],1*λ[1,2]} averaged over
the ten sets of parameters generated, and exports the results to
"BranchingAccuracybyChangeLambda.csv". Threshold 1e-20, size n=10, and complex
step size 1e-6 are used.
"""
function branchingAccuracyByChangeLambda()
    n=10
    dels=[.01,.1,1]
    L=length(dels)
    vals=zeros(L,9)
    trials=10
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    for r in 1:trials
        params=subParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        A=cumulative_inverse(birth,death,migration)
        dA_1=dAdlambda_inverse(birth,death,migration)
        dA_2=dAdlambda_iterative(birth,death,migration)
        dA_3=dAdlambda_complex(birth,death,migration)
        for d in 1:L
            change=dels[d]
            vals[d,1]=trials*change
            dlambda=migration[1,2]*change
            A1_expected=A+(dlambda*dA_1)
            A2_expected=A+(dlambda*dA_2)
            A3_expected=A+(dlambda*dA_3)
            migration[1,2]=migration[1,2]+dlambda
            migration=migration_generator(migration)
            A_actual=cumulative_inverse(birth,death,migration)
            vals[d,2]=vals[d,2]+sum(abs.(A1_expected.-A_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(A2_expected.-A_actual))/n
            vals[d,4]=vals[d,4]+sum(abs.(A3_expected.-A_actual))/n
        end
    end
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for r in 1:trials
        params=regParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        e=extinction_probability(birth,death,migration)
        de_1=de_dlambda(birth,death,migration)
        de_2=dedlambda_complex(birth,death,migration)
        de_3=dedlambda_inverse(birth,death,migration)
        drho=drho_dlambda(birth,death,migration)
        rho=find_rho(calculate_generator(birth,death,migration),n)
        dmean=dmean_ddelta(birth,death,migration)
        mean=expected_number(calculate_generator(birth,death,migration),2)
        for d in 1:L
            change=dels[d]
            vals[d,1]=trials*change
            dlambda=migration[1,2]*change
            e1_expected=e+(dlambda*de_1)
            e2_expected=e+(dlambda*de_2)
            e3_expected=e+(dlambda*de_3)
            mean_expected=mean+(dlambda*dmean)
            migration[1,2]=migration[1,2]+dlambda
            migration=migration_generator(migration)
            e_actual=extinction_probability(birth,death,migration)
            vals[d,5]=vals[d,5]+sum(abs.(e1_expected.-e_actual))/n
            vals[d,6]=vals[d,6]+sum(abs.(e2_expected.-e_actual))/n
            rho_actual=find_rho(calculate_generator(birth,death,migration),n)
            vals[d,7]=vals[d,7]+abs(rho_actual-rho_expected)
            vals[d,8]=vals[d,8]+sum(abs.(e3_expected.-e_actual))/n
            mean_actual=expected_number(calculate_generator(birth,death,migration),2)
            vals[d,9]=vals[d,9]+sum(abs.(mean_expected-mean_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("BranchingAccuracybyChangeLambda.csv",vals,',')
end

"""
    function branchingAccuracyByStepDelta

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of δ[1].  This
method compares these values across complex step=[1e9-9,1e-6,1e-3]
averaged over the ten sets of parameters generated, and exports the results to
"BranchingAccuracybyStepDelta.csv". Threshold 1e-20, size n=10, and
dδ=.1*δ[1] are used.
"""
function branchingAccuracyByStepDelta()
    n=10
    change=.1
    step=[1e-9,1e-6,1e-3]
    L=length(step)
    vals=zeros(L,3)
    trials=10
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    for r in 1:trials
        params=subParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        A=cumulative_inverse(birth,death,migration)
        ddelta=death[1]*change
        for d in 1:L
            s=step[d]
            vals[d,1]=trials*s
            dA=dAddelta_complex(birth,death,migration,1e-20,s)
            A_expected=A+(ddelta*dA)
            death[1]=death[1]+ddelta
            A_actual=cumulative_inverse(birth,death,migration)
            vals[d,2]=vals[d,2]+sum(abs.(A_expected.-A_actual))/n
        end
    end
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for r in 1:trials
        params=regParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        e=extinction_probability(birth,death,migration)
        ddelta=death[1]*change
        for d in 1:L
            s=step[d]
            vals[d,1]=trials*s
            de=deddelta_complex(birth,death,migration,1e-20,s)
            e_expected=e+(ddelta*de)
            death[1]=death[1]+ddelta
            e_actual=extinction_probability(birth,death,migration)
            vals[d,3]=vals[d,3]+sum(abs.(e_expected.-e_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("BranchingAccuracybyStepDelta.csv",vals,',')
end

"""
    function branchingAccuracyByStepBeta

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of β[1].  This
method compares these values across complex step=[1e9-9,1e-6,1e-3]
averaged over the ten sets of parameters generated, and exports the results to
"BranchingAccuracybyStepBeta.csv". Threshold 1e-20, size n=10, and
dβ=.1*β[1] are used.
"""
function branchingAccuracyByStepBeta()
    n=10
    change=.1
    step=[1e-9,1e-6,1e-3]
    L=length(step)
    vals=zeros(L,3)
    trials=10
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    for r in 1:trials
        params=subParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        A=cumulative_inverse(birth,death,migration)
        dbeta=birth[1]*change
        for d in 1:L
            s=step[d]
            vals[d,1]=trials*s
            dA=dAdbeta_complex(birth,death,migration,1e-20,s)
            A_expected=A+(dbeta*dA)
            birth[1]=birth[1]+dbeta
            A_actual=cumulative_inverse(birth,death,migration)
            vals[d,2]=vals[d,2]+sum(abs.(A_expected.-A_actual))/n
        end
    end
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for r in 1:trials
        params=regParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        e=extinction_probability(birth,death,migration)
        dbeta=birth[1]*change
        for d in 1:L
            s=step[d]
            vals[d,1]=trials*s
            de=dedbeta_complex(birth,death,migration,1e-20,s)
            e_expected=e+(dbeta*de)
            birth[1]=birth[1]+dbeta
            e_actual=extinction_probability(birth,death,migration)
            vals[d,3]=vals[d,3]+sum(abs.(e_expected.-e_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("BranchingAccuracybyStepBeta.csv",vals,',')
end

"""
    function branchingAccuracyByStepLambda

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of λ[1,2].  This
method compares these values across complex step=[1e9-9,1e-6,1e-3]
averaged over the ten sets of parameters generated, and exports the results to
"BranchingAccuracybyStepLambda.csv". Threshold 1e-20, size n=10, and
dλ=.1*λ[1,2] are used.
"""
function branchingAccuracyByStepLambda()
    n=10
    change=.1
    step=[1e-9,1e-6,1e-3]
    L=length(step)
    vals=zeros(L,3)
    trials=10
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    for r in 1:trials
        params=subParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        A=cumulative_inverse(birth,death,migration)
        dlambda=migration[1,2]*change
        for d in 1:L
            s=step[d]
            vals[d,1]=trials*s
            dA=dAdlambda_complex(birth,death,migration,1e-20,s)
            A_expected=A+(dlambda*dA)
            migration[1,2]=migration[1,2]+dlambda
            migration=migration_generator(migration)
            A_actual=cumulative_inverse(birth,death,migration)
            vals[d,2]=vals[d,2]+sum(abs.(A_expected.-A_actual))/n
        end
    end
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for r in 1:trials
        params=regParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        e=extinction_probability(birth,death,migration)
        dlambda=migration[1,2]*change
        for d in 1:L
            s=step[d]
            vals[d,1]=trials*s
            de=dedlambda_complex(birth,death,migration,1e-20,s)
            e_expected=e+(dlambda*de)
            migration[1,2]=migration[1,2]+dlambda
            migration=migration_generator(migration)
            e_actual=extinction_probability(birth,death,migration)
            vals[d,3]=vals[d,3]+sum(abs.(e_expected.-e_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("BranchingAccuracybyStepLambda.csv",vals,',')
end

"""
    function branchingAccuracyByThresholdDelta

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of δ[1].  This
method compares these values across convergence threshold=[1e-16,1e-12,1e-8]
averaged over the ten sets of parameters generated, and exports the results to
"BranchingAccuracybyThresholdDelta.csv". Complex step 1e-6, size n=10, and
dδ=.1*δ[1] are used.
"""
function branchingAccuracyByThresholdDelta()
    n=10
    thresh=[1e-16,1e-12,1e-8]
    change=.1
    L=length(thresh)
    vals=zeros(L,6)
    trials=10
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    for r in 1:trials
        params=subParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        A=cumulative_inverse(birth,death,migration)
        ddelta=death[1]*change
        for d in 1:L
            t=thresh[d]
            vals[d,1]=trials*t
            dA_1=dAddelta_iterative(birth,death,migration,t)
            dA_2=dAddelta_complex(birth,death,migration,t)
            A1_expected=A+(ddelta*dA_1)
            A2_expected=A+(ddelta*dA_2)
            death[1]=death[1]+ddelta
            A_actual=cumulative_inverse(birth,death,migration)
            vals[d,2]=vals[d,2]+sum(abs.(A1_expected.-A_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(A2_expected.-A_actual))/n
        end
    end
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for r in 1:trials
        params=regParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        e=extinction_probability(birth,death,migration)
        dmean=dmean_ddelta(birth,death,migration)
        mean=expected_number(calculate_generator(birth,death,migration),2)
        ddelta=death[1]*change
        for d in 1:L
            t=thresh[d]
            vals[d,1]=trials*t
            de_1=de_ddelta(birth,death,migration,t)
            de_2=deddelta_complex(birth,death,migration,t)
            e1_expected=e+(ddelta*de_1)
            e2_expected=e+(ddelta*de_2)
            mean_expected=mean+(ddelta*dmean)
            death[1]=death[1]+ddelta
            e_actual=extinction_probability(birth,death,migration)
            vals[d,4]=vals[d,4]+sum(abs.(e1_expected.-e_actual))/n
            vals[d,5]=vals[d,5]+sum(abs.(e2_expected.-e_actual))/n
            mean_actual=expected_number(calculate_generator(birth,death,migration),2)
            vals[d,6]=vals[d,6]+sum(abs.(mean_expected-mean_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("BranchingAccuracybyThresholdDelta.csv",vals,',')
end

"""
    function branchingAccuracyByThresholdBeta

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of β[1].  This
method compares these values across convergence threshold=[1e-16,1e-12,1e-8]
averaged over the ten sets of parameters generated, and exports the results to
"BranchingAccuracybyThresholdBeta.csv". Complex step 1e-6, size n=10, and
dβ=.1*β[1] are used.
"""
function branchingAccuracyByThresholdBeta()
    n=10
    thresh=[1e-16,1e-12,1e-8]
    change=.1
    L=length(thresh)
    vals=zeros(L,6)
    trials=10
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    for r in 1:trials
        params=subParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        A=cumulative_inverse(birth,death,migration)
        dbeta=birth[1]*change
        for d in 1:L
            t=thresh[d]
            vals[d,1]=trials*t
            dA_1=dAdbeta_iterative(birth,death,migration,t)
            dA_2=dAdbeta_complex(birth,death,migration,t)
            A1_expected=A+(dbeta*dA_1)
            A2_expected=A+(dbeta*dA_2)
            birth[1]=birth[1]+dbeta
            A_actual=cumulative_inverse(birth,death,migration)
            vals[d,2]=vals[d,2]+sum(abs.(A1_expected.-A_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(A2_expected.-A_actual))/n
        end
    end
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for r in 1:trials
        params=regParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        e=extinction_probability(birth,death,migration)
        dmean=dmean_ddelta(birth,death,migration)
        mean=expected_number(calculate_generator(birth,death,migration),2)
        dbeta=birth[1]*change
        for d in 1:L
            t=thresh[d]
            vals[d,1]=trials*t
            de_1=de_dbeta(birth,death,migration,t)
            de_2=dedbeta_complex(birth,death,migration,t)
            e1_expected=e+(dbeta*de_1)
            e2_expected=e+(dbeta*de_2)
            mean_expected=mean+(dbeta*dmean)
            birth[1]=birth[1]+dbeta
            e_actual=extinction_probability(birth,death,migration)
            vals[d,4]=vals[d,4]+sum(abs.(e1_expected.-e_actual))/n
            vals[d,5]=vals[d,5]+sum(abs.(e2_expected.-e_actual))/n
            mean_actual=expected_number(calculate_generator(birth,death,migration),2)
            vals[d,6]=vals[d,6]+sum(abs.(mean_expected-mean_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("BranchingAccuracybyThresholdBeta.csv",vals,',')
end

"""
    function branchingAccuracyByThresholdLambda

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of λ[1,2].
This method compares these values across convergence
threshold=[1e-16,1e-12,1e-8] averaged over the ten sets of parameters generated,
and exports the results to "BranchingAccuracybyThresholdLambda.csv". Complex
step 1e-6, size n=10, and dλ=.1*λ[1,2] are used.
"""
function branchingAccuracyByThresholdLambda()
    n=10
    thresh=[1e-16,1e-12,1e-8]
    change=.1
    L=length(thresh)
    vals=zeros(L,6)
    trials=10
    subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
    for r in 1:trials
        params=subParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        A=cumulative_inverse(birth,death,migration)
        dlambda=migration[1,2]*change
        for d in 1:L
            t=thresh[d]
            vals[d,1]=trials*t
            dA_1=dAdlambda_iterative(birth,death,migration,t)
            dA_2=dAdlambda_complex(birth,death,migration,t)
            A1_expected=A+(dlambda*dA_1)
            A2_expected=A+(dlambda*dA_2)
            migration[1,2]=migration[1,2]+dlambda
            migration=migration_generator(migration)
            A_actual=cumulative_inverse(birth,death,migration)
            vals[d,2]=vals[d,2]+sum(abs.(A1_expected.-A_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(A2_expected.-A_actual))/n
        end
    end
    regParams=readdlm(string(n,"RealisticParams.csv"),',')
    for r in 1:trials
        params=regParams[n*(r-1)+1:n*r,:]
        birth=params[:,1]
        death=params[:,2]
        migration=params[:,3:end]
        #e=extinction_probability(birth,death,migration)
        dmean=dmean_ddelta(birth,death,migration)
        mean=expected_number(calculate_generator(birth,death,migration),2)
        dlambda=migration[1,2]*change
        for d in 1:L
            t=thresh[d]
            vals[d,1]=trials*t
            de_1=de_dlambda(birth,death,migration,t)
            de_2=dedlambda_complex(birth,death,migration,t)
            e1_expected=e+(dlambda*de_1)
            e2_expected=e+(dlambda*de_2)
            mean_expected=mean+(dlambda*dmean)
            migration[1,2]=migration[1,2]+dlambda
            migration=migration_generator(migration)
            e_actual=extinction_probability(birth,death,migration)
            vals[d,4]=vals[d,4]+sum(abs.(e1_expected.-e_actual))/n
            vals[d,5]=vals[d,5]+sum(abs.(e2_expected.-e_actual))/n
            mean_actual=expected_number(calculate_generator(birth,death,migration),2)
            vals[d,6]=vals[d,6]+sum(abs.(mean_expected-mean_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("BranchingAccuracybyThresholdLambda.csv",vals,',')
end

"""
    function branchingAccuracyBySizeDelta

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of δ[1].  This
method compares these values across n=[10,100,1000] averaged over the ten sets
of parameters generated, and exports the results to
"BranchingAccuracybySizeDelta.csv". Complex step 1e-6, threshold 1e-20, and
dδ=.1*δ[1] are used.
"""
function branchingAccuracyBySizeDelta()
    N=[10,100,1000]
    change=.1
    L=length(N)
    vals=zeros(L,9)
    trials=10
    iter=0
    for n in N
        iter=iter+1
        vals[iter,1]=trials*n
        subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
        for r in 1:trials
            params=subParams[n*(r-1)+1:n*r,:]
            birth=params[:,1]
            death=params[:,2]
            migration=params[:,3:end]
            A=cumulative_inverse(birth,death,migration)
            dA_1=dAddelta_inverse(birth,death,migration)
            dA_2=dAddelta_iterative(birth,death,migration)
            dA_3=dAddelta_complex(birth,death,migration)
            ddelta=death[1]*change
            A1_expected=A+(ddelta*dA_1)
            A2_expected=A+(ddelta*dA_2)
            A3_expected=A+(ddelta*dA_3)
            death[1]=death[1]+ddelta
            A_actual=cumulative_inverse(birth,death,migration)
            vals[iter,2]=vals[iter,2]+sum(abs.(A1_expected.-A_actual))/n
            vals[iter,3]=vals[iter,3]+sum(abs.(A2_expected.-A_actual))/n
            vals[iter,4]=vals[iter,4]+sum(abs.(A3_expected.-A_actual))/n
        end
        regParams=readdlm(string(n,"RealisticParams.csv"),',')
        for r in 1:trials
            params=regParams[n*(r-1)+1:n*r,:]
            birth=params[:,1]
            death=params[:,2]
            migration=params[:,3:end]
            e=extinction_probability(birth,death,migration)
            de_1=de_ddelta(birth,death,migration)
            de_2=deddelta_complex(birth,death,migration)
            de_3=deddelta_complex(birth,death,migration)
            drho=drho_ddelta(birth,death,migration)
            rho=find_rho(calculate_generator(birth,death,migration),n)
            dmean=dmean_ddelta(birth,death,migration)
            mean=expected_number(calculate_generator(birth,death,migration),2)
            ddelta=death[1]*change
            e1_expected=e+(ddelta*de_1)
            e2_expected=e+(ddelta*de_2)
            e3_expected=e+(ddelta*de_3)
            mean_expected=mean+(ddelta*dmean)
            death[1]=death[1]+ddelta
            e_actual=extinction_probability(birth,death,migration)
            vals[iter,5]=vals[iter,5]+sum(abs.(e1_expected.-e_actual))/n
            vals[iter,6]=vals[iter,6]+sum(abs.(e2_expected.-e_actual))/n
            rho_actual=find_rho(calculate_generator(birth,death,migration),n)
            vals[iter,7]=vals[iter,7]+abs(rho_actual-rho_expected)
            vals[iter,8]=vals[iter,8]+sum(abs.(e3_expected.-e_actual))/n
            mean_actual=expected_number(calculate_generator(birth,death,migration),2)
            vals[iter,9]=vals[iter,9]+sum(abs.(mean_expected-mean_actual))/n
        end
        println(vals[iter,:]./trials)
    end
    vals=vals./trials
    writedlm("BranchingAccuracybySizeDelta.csv",vals,',')
end

"""
    function branchingAccuracyBySizeBeta

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of β[1].  This
method compares these values across n=[10,100,1000] averaged over the ten sets
of parameters generated, and exports the results to
"BranchingAccuracybySizeBeta.csv". Complex step 1e-6, threshold 1e-20, and
dβ=.1*β[1] are used.
"""
function branchingAccuracyBySizeBeta(change)
    N=[10,100,1000]
    #N = [10]
    change=.01
    L=length(N)
    vals=zeros(L,9)
    #trials=10
    trials = 1
    iter=0
    for n in N
        iter=iter+1
        vals[iter,1]=trials*n
        subParams=readdlm(string("../data/",n,"RealisticSubCriticalParams.csv"),',')
        for r in 1:trials
            params=subParams[n*(r-1)+1:n*r,:]
            birth=params[:,1]
            death=params[:,2]
            migration=params[:,3:end]
            A=cumulative_inverse(birth,death,migration)
            dA_1=dAdbeta_inverse(birth,death,migration, 1e-18, 1e-12)
            dA_2=dAdbeta_complexinverse(birth,death,migration, 1e-18, 1e-12)
            #dA_3=dAdbeta_iterative(birth,death,migration)
            dbeta=birth[1]*change
            A1_expected=A+(dbeta*dA_1)
            A2_expected=A+(dbeta*dA_2)
            #A3_expected=A+(dbeta*dA_3)
            birth[1]=birth[1]+dbeta
            A_actual=cumulative_inverse(birth,death,migration)
            vals[iter,2]=vals[iter,2]+ sqrt(sum((A1_expected.-A_actual).^2))
            vals[iter,3]=vals[iter,3]+ sqrt(sum((A2_expected.-A_actual).^2))
            #vals[iter,4]=vals[iter,4]+ sqrt(sum((A3_expected.-A_actual).^2))
        end
        regParams=readdlm(string("../data/",n, "RealisticParams.csv"),',')
        for r in 1:trials
            params=regParams[n*(r-1)+1:n*r,:]
            birth=params[:,1]
            death=params[:,2]
            migration=params[:,3:end]
            e=extinction_probability(birth,death,migration)
            de_1=de_dbeta(birth,death,migration, 1e-18, 1e-12)
            de_2=dedbeta_complex(birth,death,migration, 1e-18, 1e-12)
            #de_3=dedbeta_inverse(birth,death,migration)
            #drho=drho_dbeta(birth,death,migration)
            #rho=find_rho(calculate_generator(birth,death,migration),n)
            dbeta=birth[1]*change
            #dmean=dmean_ddelta(birth,death,migration)
            #mean=expected_number(calculate_generator(birth,death,migration),2)
            #mean_expected=mean+(dbeta*dmean)
            e1_expected=e+(dbeta*de_1)
            e2_expected=e+(dbeta*de_2)
            #e3_expected=e+(dbeta*de_3)
            birth[1]=birth[1]+dbeta
            #mean_actual=expected_number(calculate_generator(birth,death,migration),2)
            #vals[iter,9]=vals[iter,9]+sum(abs.(mean_expected-mean_actual))/n
            e_actual=extinction_probability(birth,death,migration)
            vals[iter,5]=vals[iter,5]+sqrt(sum((e1_expected.-e_actual).^2))
            vals[iter,6]=vals[iter,6]+sqrt(sum((e2_expected.-e_actual).^2))
            #rho_actual=find_rho(calculate_generator(birth,death,migration),n)
            #vals[iter,7]=vals[iter,7]+abs(rho_actual-rho_expected)
            #vals[iter,7]=vals[iter,7]+sqrt(sum((e3_expected.-e_actual).^2))
        end
        #println(vals[iter,:]./trials)
    end
    vals=vals./trials
    #writedlm("BranchingAccuracybySizeBeta.csv",vals,',')
    return vals
end

"""
    function branchingAccuracyBySizeLambda

This function calculates loss of accuracy for the branching process methods,
comparing the expected values of A and e using the finite difference formulas
with the actual values of A and e with the different value of λ[1,2].
This method compares these values across n=[10,100,1000] averaged over the ten
sets of parameters generated, and exports the results to
"BranchingAccuracybySizeLambda.csv". Complex step 1e-6, threshold 1e-20, and
dλ=.1*λ[1,2] are used.
"""
function branchingAccuracyBySizeLambda()
    N=[10,100,1000]
    change=.1
    L=length(N)
    vals=zeros(L,9)
    trials=10
    iter=0
    for n in N
        iter=iter+1
        vals[iter,1]=trials*n
        subParams=readdlm(string(n,"RealisticSubCriticalParams.csv"),',')
        for r in 1:trials
            params=subParams[n*(r-1)+1:n*r,:]
            birth=params[:,1]
            death=params[:,2]
            migration=params[:,3:end]
            A=cumulative_inverse(birth,death,migration)
            dA_1=dAdlambda_inverse(birth,death,migration)
            dA_2=dAdlambda_iterative(birth,death,migration)
            dA_3=dAdlambda_complex(birth,death,migration)
            dlambda=migration[1,2]*change
            A1_expected=A+(dlambda*dA_1)
            A2_expected=A+(dlambda*dA_2)
            A3_expected=A+(dlambda*dA_3)
            migration[1,2]=migration[1,2]+dlambda
            migration=migration_generator(migration)
            A_actual=cumulative_inverse(birth,death,migration)
            vals[iter,2]=vals[iter,2]+sum(abs.(A1_expected.-A_actual))/n
            vals[iter,3]=vals[iter,3]+sum(abs.(A2_expected.-A_actual))/n
            vals[iter,4]=vals[iter,4]+sum(abs.(A3_expected.-A_actual))/n
        end
        regParams=readdlm(string(n,"RealisticParams.csv"),',')
        for r in 1:trials
            params=regParams[n*(r-1)+1:n*r,:]
            birth=params[:,1]
            death=params[:,2]
            migration=params[:,3:end]
            e=extinction_probability(birth,death,migration)
            de_1=de_dlambda(birth,death,migration)
            de_2=dedlambda_complex(birth,death,migration)
            de_3=dedlambda_inverse(birth,death,migration)
            drho=drho_dlambda(birth,death,migration)
            rho=find_rho(calculate_generator(birth,death,migration),n)
            dmean=dmean_ddelta(birth,death,migration)
            mean=expected_number(calculate_generator(birth,death,migration),2)
            dlambda=migration[1,2]*change
            e1_expected=e+(dlambda*de_1)
            e2_expected=e+(dlambda*de_2)
            e3_expected=e+(dlambda*de_3)
            mean_expected=mean+(dlambda*dmean)
            migration[1,2]=migration[1,2]+dlambda
            migration=migration_generator(migration)
            e_actual=extinction_probability(birth,death,migration)
            vals[iter,5]=vals[iter,5]+sum(abs.(e1_expected.-e_actual))/n
            vals[iter,6]=vals[iter,6]+sum(abs.(e2_expected.-e_actual))/n
            rho_actual=find_rho(calculate_generator(birth,death,migration),n)
            vals[iter,7]=vals[iter,7]+abs(rho_actual-rho_expected)
            vals[iter,8]=vals[iter,8]+sum(abs.(e3_expected.-e_actual))/n
            mean_actual=expected_number(calculate_generator(birth,death,migration),2)
            vals[iter,9]=vals[iter,9]+sum(abs.(mean_expected-mean_actual))/n
        end
        println(vals[iter,:]./trials)
    end
    vals=vals./trials
    writedlm("BranchingAccuracybySizeLambda.csv",vals,',')
end

#=
Benchmarking Methods for the SIR Model
=#

"""
    function benchmarkSIR_bySizeDelta()

This function calculates time, memory, and loss of precision for the SIR model
methods over N=10,100,1000 averaged over the ten sets of
parameters generated, and exports the results to "SIRbySizeDelta.csv".  The
derivatives are calculated with θ=δ. Step 1e-6 is used where applicable.
"""
function benchmarkSIR_bySizeDelta()
    params=readdlm("RealisticSIRParams.csv",',')
    params_s=convert(Array{Float32,2},params)
    trials=10
    N=[10,100,1000]
    F=[meannumber,meantime,dm_ddelta,dmddelta_complex,dt_ddelta,dtddelta_complex]
    iter=0
    vals=zeros(length(N)*length(F),5)
    for f in F
        for n in N
            iter=iter+1
            vals[iter,2]=n*trials
            for r in 1:trials
                delta=params[1,r]
                delta_s=params_s[1,r]
                eta=params[2,r]
                eta_s=params_s[2,r]
                test=median(@benchmark $f($n,$delta,$eta))
                diff=sum(abs.(f(n,delta,eta).-f(n,delta_s,eta_s,Float32)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("SIRbySizeDelta.csv",vals,',')
    return vals
end

"""
    function benchmarkSIR_bySizeEta()

This function calculates time, memory, and loss of precision for the SIR model
methods over N=10,100,1000 averaged over the ten sets of parameters generated,
and exports the results to "SIRbySizeEta.csv".  The derivatives are calculated
with θ=δ. Step 1e-6 is used where applicable.
"""
function benchmarkSIR_bySizeEta()
    params=readdlm("../data/RealisticSIRParams.csv",',')
    params_s=convert(Array{Float32,2},params)
    #trials=10
    trials = 1
    N=[10,100,1000]
    #N = [10]
    F=[dm_deta,dmdeta_complex,dt_deta,dtdeta_complex]
    iter=0
    vals=zeros(length(N)*length(F),5)
    for f in F
        for n in N
            iter=iter+1
            vals[iter,2]=n*trials
            for r in 1:trials
                delta=params[1,r]
                #delta_s=params_s[1,r]
                eta=params[2,r]
                #eta_s=params_s[2,r]
                test=median(@benchmark $f($n,$delta,$eta, 1e-12))
                #diff=sum(abs.(f(n,delta,eta).-f(n,delta_s,eta_s,Float32)))/n
                vals[iter,3]=vals[iter,3]+test.time
                #vals[iter,4]=vals[iter,4]+test.memory
                #vals[iter,5]=vals[iter,5]+diff
            end
            #println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    #writedlm("SIRbySizeEta.csv",vals,',')
    return vals
end

"""
    function benchmarkSIR_byStepDelta()

This function calculates time, memory, and loss of precision for the SIR model
methods over step=1e-3,1e-6,1e-9 averaged over the ten sets
of  parameters generated, and exports the results to "SIRbyStepDelta.csv".  The
derivatives are calculated with θ=δ. Size n=100 is used.
"""
function benchmarkSIR_byStepDelta()
    params=readdlm("RealisticSIRParams.csv",',')
    params_s=convert(Array{Float32,2},params)
    trials=10
    n=100
    eps=[1e-3,1e-6,1e-9]
    F=[dmddelta_complex,dtddelta_complex]
    iter=0
    vals=zeros(length(eps)*length(F),5)
    for f in F
        for step in eps
            iter=iter+1
            vals[iter,2]=step*trials
            step_s=convert(Float32,step)
            for r in 1:trials
                delta=params[1,r]
                delta_s=params_s[1,r]
                eta=params[2,r]
                eta_s=params_s[2,r]
                test=median(@benchmark $f($n,$delta,$eta,Float64,$step))
                diff=sum(abs.(f(n,delta,eta,Float64,step).-f(n,delta_s,eta_s,Float32,step_s)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("SIRbyStepDelta.csv",vals,',')
    return vals
end

"""
    function benchmarkSIR_byStepEta()

This function calculates time, memory, and loss of precision for the SIR model
methods over step=1e-3,1e-6,1e-9 averaged over the ten sets of  parameters
generated, and exports the results to "SIRbyStepEta.csv".  The derivatives are
calculated with θ=η. Size n=100 is used.
"""
function benchmarkSIR_byStepEta()
    params=readdlm("RealisticSIRParams.csv",',')
    params_s=convert(Array{Float32,2},params)
    trials=10
    n=100
    eps=[1e-3,1e-6,1e-9]
    F=[dmdeta_complex,dtdeta_complex]
    iter=0
    vals=zeros(length(eps)*length(F),5)
    for f in F
        for step in eps
            iter=iter+1
            vals[iter,2]=step*trials
            step_s=convert(Float32,step)
            for r in 1:trials
                delta=params[1,r]
                delta_s=params_s[1,r]
                eta=params[2,r]
                eta_s=params_s[2,r]
                test=median(@benchmark $f($n,$delta,$eta,Float64,$step))
                diff=sum(abs.(f(n,delta,eta,Float64,step).-f(n,delta_s,eta_s,Float32,step_s)))/n
                vals[iter,3]=vals[iter,3]+test.time
                vals[iter,4]=vals[iter,4]+test.memory
                vals[iter,5]=vals[iter,5]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("SIRbyStepEta.csv",vals,',')
    return vals
end

"""
    function compareSIR_bySizeDelta

This function calculates loss of accuracy for the SIR model methods, comparing
outomes of iterative and complex methods for
N=10,100,1000 averaged over the ten sets of parameters generated,
and exports the results to "SIRComparebySizeDelta.csv".  The derivatives are
calculated with θ=δ.  Step=1e-6 is used where applicable.
"""
function compareSIR_bySizeDelta()
    params=readdlm("RealisticSIRParams.csv",',')
    trials=10
    N=[10,100,1000]
    F1=[dm_ddelta,dt_ddelta]
    F2=[dmddelta_complex,dtddelta_complex]
    num_comp=length(F1)
    iter=0
    vals=zeros(length(N)*num_comp,3)
    for n in N
        for a=1:num_comp
            iter=iter+1
            vals[iter,2]=n*trials
            f1=F1[a]
            f2=F2[a]
            for r in 1:trials
                delta=params[1,r]
                eta=params[2,r]
                diff=sum(abs.(f1(n,delta,eta).-f2(n,delta,eta)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("SIRComparebySizeDelta.csv",vals,',')
    return vals
end

"""
    function compareSIR_bySizeEta

This function calculates loss of accuracy for the SIR model methods, comparing
outomes of iterative and complex methods for N=10,100,1000 averaged over the
ten sets of parameters generated, and exports the results to
"SIRComparebySizeEta.csv".  The derivatives are calculated with θ=η.
Step=1e-6 is used where applicable.
"""
function compareSIR_bySizeEta()
    params=readdlm("RealisticSIRParams.csv",',')
    trials=10
    N=[10,100,1000]
    F1=[dm_deta,dt_deta]
    F2=[dmdeta_complex,dtdeta_complex]
    num_comp=length(F1)
    iter=0
    vals=zeros(length(N)*num_comp,3)
    for n in N
        for a=1:num_comp
            iter=iter+1
            vals[iter,2]=n*trials
            f1=F1[a]
            f2=F2[a]
            for r in 1:trials
                delta=params[1,r]
                eta=params[2,r]
                diff=sum(abs.(f1(n,delta,eta).-f2(n,delta,eta)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("SIRComparebySizeEta.csv",vals,',')
    return vals
end

"""
    function compareSIR_byStepDelta

This function calculates loss of accuracy for the SIR model methods, comparing
outomes of iterative and complex methods for
step=1e-3,1e-6,1e-9 averaged over the ten sets of
parameters generated, and exports the results to "SIRComparebyStepDelta.csv".
The derivatives are calculated with θ=δ.  Size N=100 is used.
"""
function compareSIR_byStepDelta()
    params=readdlm("RealisticSIRParams.csv",',')
    trials=10
    n=100
    eps=[1e-3,1e-6,1e-9]
    F1=[dm_ddelta,dt_ddelta]
    F2=[dmddelta_complex,dtddelta_complex]
    num_comp=length(F1)
    iter=0
    vals=zeros(length(eps)*num_comp,3)
    for step in eps
        for a=1:num_comp
            iter=iter+1
            vals[iter,2]=step*trials
            f1=F1[a]
            f2=F2[a]
            for r in 1:trials
                delta=params[1,r]
                eta=params[2,r]
                diff=sum(abs.(f1(n,delta,eta,step,Float64).-f2(n,delta,eta,Float64,step)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("SIRComparebyStepDelta.csv",vals,',')
    return vals
end

"""
    function compareSIR_byStepEta

This function calculates loss of accuracy for the SIR model methods, comparing
outomes of iterative and complex methods for step=1e-3,1e-6,1e-9 averaged over
the ten sets of parameters generated, and exports the results to
"SIRComparebyStepEta.csv".  The derivatives are calculated with θ=η.
Size N=100 is used.
"""
function compareSIR_byStepEta()
    params=readdlm("RealisticSIRParams.csv",',')
    trials=10
    n=100
    eps=[1e-3,1e-6,1e-9]
    F1=[dm_deta,dt_deta]
    F2=[dmdeta_complex,dtdeta_complex]
    num_comp=length(F1)
    iter=0
    vals=zeros(length(eps)*num_comp,3)
    for step in eps
        for a=1:num_comp
            iter=iter+1
            vals[iter,2]=step*trials
            f1=F1[a]
            f2=F2[a]
            for r in 1:trials
                delta=params[1,r]
                eta=params[2,r]
                diff=sum(abs.(f1(n,delta,eta,step,Float64).-f2(n,delta,eta,Float64,step)))/n
                vals[iter,3]=vals[iter,3]+diff
            end
            println(vals[iter,:]./trials)
        end
    end
    vals=vals./trials
    writedlm("SIRComparebyStepEta.csv",vals,',')
    return vals
end

"""
    function SIRAccuracybyChangeDelta

This function calculates loss of accuracy for the SIR model methods, comparing
the expected values of M and T using the finite difference formulas with the
actual values of M and T with the different value of δ.  This method
compares these values across dδ=[.01*δ,.1*δ,1*δ] averaged over
the ten sets of parameters generated, and exports the results to
"SIRAccuracybyChangeDelta.csv". Size N=100 and complex step 1e-6 are used.
"""
function SIRAccuracyByChangeDelta()
    n=100
    dels=[.01,.1,1]
    L=length(dels)
    vals=zeros(L,5)
    trials=10
    params=readdlm("RealisticSIRParams.csv",',')
    for r in 1:trials
        delta=params[1,r]
        eta=params[2,r]
        M=meannumber(n,delta,eta)
        dM_1=dm_ddelta(n,delta,eta)
        dM_2=dmddelta_complex(n,delta,eta)
        T=meantime(n,delta,eta)
        dT_1=dt_ddelta(n,delta,eta)
        dT_2=dtddelta_complex(n,delta,eta)
        for d in 1:L
            change=dels[d]
            vals[d,1]=trials*change
            ddelta=delta*change
            M1_expected=M+(ddelta*dM_1)
            M2_expected=M+(ddelta*dM_2)
            T1_expected=T+(ddelta*dT_1)
            T2_expected=T+(ddelta*dT_2)
            delta=delta+ddelta
            M_actual=meannumber(n,delta,eta)
            T_actual=meantime(n,delta,eta)
            vals[d,2]=vals[d,2]+sum(abs.(M1_expected.-M_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(M2_expected.-M_actual))/n
            vals[d,4]=vals[d,4]+sum(abs.(T1_expected.-T_actual))/n
            vals[d,5]=vals[d,5]+sum(abs.(T2_expected.-T_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("SIRAccuracybyChangeDelta.csv",vals,',')
end

"""
    function SIRAccuracybyChangeEta

This function calculates loss of accuracy for the SIR model methods, comparing
the expected values of M and T using the finite difference formulas with the
actual values of M and T with the different value of η.  This method
compares these values across deta=[.01*η,.1*η,1*η] averaged over
the ten sets of parameters generated, and exports the results to
"SIRAccuracybyChangeEta.csv". Size N=100 and complex step 1e-6 are used.
"""
function SIRAccuracyByChangeEta()
    n=100
    dels=[.01,.1,1]
    L=length(dels)
    vals=zeros(L,5)
    trials=10
    params=readdlm("RealisticSIRParams.csv",',')
    for r in 1:trials
        delta=params[1,r]
        eta=params[2,r]
        M=meannumber(n,delta,eta)
        dM_1=dm_deta(n,delta,eta)
        dM_2=dmdeta_complex(n,delta,eta)
        T=meantime(n,delta,eta)
        dT_1=dt_deta(n,delta,eta)
        dT_2=dtdeta_complex(n,delta,eta)
        for d in 1:L
            change=dels[d]
            vals[d,1]=trials*change
            deta=eta*change
            M1_expected=M+(deta*dM_1)
            M2_expected=M+(deta*dM_2)
            T1_expected=T+(deta*dT_1)
            T2_expected=T+(deta*dT_2)
            eta=eta+deta
            M_actual=meannumber(n,delta,eta)
            T_actual=meantime(n,delta,eta)
            vals[d,2]=vals[d,2]+sum(abs.(M1_expected.-M_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(M2_expected.-M_actual))/n
            vals[d,4]=vals[d,4]+sum(abs.(T1_expected.-T_actual))/n
            vals[d,5]=vals[d,5]+sum(abs.(T2_expected.-T_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("SIRAccuracybyChangeEta.csv",vals,',')
end

"""
    function SIRAccuracybyStepDelta

This function calculates loss of accuracy for the SIR model methods, comparing
the expected values of M and T using the finite difference formulas with the
actual values of M and T with the different value of δ.  This method
compares these values across complex step=[1e-9,1e-6,1e-3] averaged over the
ten sets of parameters generated, and exports the results to
"SIRAccuracybyStepDelta.csv". Size N=100 and dδ=.1*δ are used.
"""
function SIRAccuracyByStepDelta()
    n=100
    change=.1
    step=[1e-9,1e-6,1e-3]
    L=length(step)
    vals=zeros(L,3)
    trials=10
    params=readdlm("RealisticSIRParams.csv",',')
    for r in 1:trials
        delta=params[1,r]
        eta=params[2,r]
        M=meannumber(n,delta,eta)
        T=meantime(n,delta,eta)
        deta=eta*change
        for d in 1:L
            s=step[d]
            vals[d,1]=trials*s
            dT=dtddelta_complex(n,delta,eta,Float64,s)
            dM=dmddelta_complex(n,delta,eta,Float64,s)
            M_expected=M+(ddelta*dM)
            T_expected=T+(ddelta*dT)
            delta=delta+ddelta
            M_actual=meannumber(n,delta,eta)
            T_actual=meantime(n,delta,eta)
            vals[d,2]=vals[d,2]+sum(abs.(M_expected.-M_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(T_expected.-T_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("SIRAccuracybyStepDelta.csv",vals,',')
end

"""
    function SIRAccuracybyStepEta

This function calculates loss of accuracy for the SIR model methods, comparing
the expected values of M and T using the finite difference formulas with the
actual values of M and T with the different value of η.  This method
compares these values across complex step=[1e-9,1e-6,1e-3] averaged over the
ten sets of parameters generated, and exports the results to
"SIRAccuracybyStepEta.csv". Size N=100 and deta=.1*η are used.
"""
function SIRAccuracyByStepEta()
    n=100
    change=.1
    step=[1e-9,1e-6,1e-3]
    L=length(step)
    vals=zeros(L,3)
    trials=10
    params=readdlm("RealisticSIRParams.csv",',')
    for r in 1:trials
        delta=params[1,r]
        eta=params[2,r]
        M=meannumber(n,delta,eta)
        T=meantime(n,delta,eta)
        deta=eta*change
        for d in 1:L
            s=step[d]
            vals[d,1]=trials*s
            dT=dtdeta_complex(n,delta,eta,Float64,s)
            dM=dmdeta_complex(n,delta,eta,Float64,s)
            M_expected=M+(deta*dM)
            T_expected=T+(deta*dT)
            eta=eta+deta
            M_actual=meannumber(n,delta,eta)
            T_actual=meantime(n,delta,eta)
            vals[d,2]=vals[d,2]+sum(abs.(M_expected.-M_actual))/n
            vals[d,3]=vals[d,3]+sum(abs.(T_expected.-T_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    writedlm("SIRAccuracybyStepEta.csv",vals,',')
end

"""
    function SIRAccuracybySizeDelta

This function calculates loss of accuracy for the SIR model methods, comparing
the expected values of M and T using the finite difference formulas with the
actual values of M and T with the different value of δ.  This method
compares these values across n=[10,100,1000] averaged over the
ten sets of parameters generated, and exports the results to
"SIRAccuracybySizeDelta.csv". Complex step 1e-6 and dδ=.1*δ are used.
"""
function SIRAccuracyBySizeDelta(change)
    #N=[10,100,1000]
    N = [10]
    L=length(N)
    vals=zeros(L,5)
    #trials=10
    trials = 1
    params=readdlm("RealisticSIRParams.csv",',')
    for r in 1:trials
        delta=params[1,r]
        eta=params[2,r]
        ddelta=delta*change
        for d in 1:L
            n=N[d]
            vals[d,1]=trials*n
            M=meannumber(n,delta,eta)
            dM_1=dm_ddelta(n,delta,eta)
            dM_2=dmddelta_complex(n,delta,eta, 1e-12)
            T=meantime(n,delta,eta)
            dT_1=dt_ddelta(n,delta,eta)
            dT_2=dtddelta_complex(n,delta,eta, 1e-12)
            M1_expected=M+(ddelta*dM_1)
            M2_expected=M+(ddelta*dM_2)
            T1_expected=T+(ddelta*dT_1)
            T2_expected=T+(ddelta*dT_2)
            delta=delta+ddelta
            M_actual=meannumber(n,delta,eta)
            T_actual=meantime(n,delta,eta)
            #vals[d,2]=vals[d,2]+sum((abs.(M1_expected.-M_actual).^2)/n
            vals[d,3]=vals[d,3]+sum(abs.(M2_expected.-M_actual))/n
            vals[d,4]=vals[d,4]+sum(abs.(T1_expected.-T_actual))/n
            vals[d,5]=vals[d,5]+sum(abs.(T2_expected.-T_actual))/n
        end
    end
    vals=vals./trials
    println(vals)
    #writedlm("SIRAccuracybySizeDelta.csv",vals,',')
    return vals
end

"""
    function SIRAccuracybySizeEta

This function calculates loss of accuracy for the SIR model methods, comparing
the expected values of M and T using the finite difference formulas with the
actual values of M and T with the different value of η.  This method
compares these values across n=[10,100,1000] averaged over the
ten sets of parameters generated, and exports the results to
"SIRAccuracybySizeEta.csv". Complex step 1e-6 and deta=.1*η are used.
"""
function SIRAccuracyBySizeEta(change)
    N=[10,100,100]
    #N = [10]
    L=length(N)
    vals=zeros(L,5)
    #trials=10
    trials = 1
    params=readdlm("../data/RealisticSIRParams.csv",',')
    for r in 1:trials
        delta=params[1,r]
        eta=params[2,r]
        deta=eta*change
        for d in 1:L
            n=N[d]
            vals[d,1]=trials*n
            M=meannumber(n,delta,eta)
            dM_1=dm_deta(n,delta,eta)
            dM_2=dmdeta_complex(n,delta,eta, 1e-12)
            T=meantime(n,delta,eta)
            dT_1=dt_deta(n,delta,eta)
            dT_2=dtdeta_complex(n,delta,eta, 1e-12)
            M1_expected=M+(deta*dM_1)
            M2_expected=M+(deta*dM_2)
            T1_expected=T+(deta*dT_1)
            T2_expected=T+(deta*dT_2)
            eta=eta+deta
            M_actual=meannumber(n,delta,eta)
            T_actual=meantime(n,delta,eta)
            vals[d,2]=vals[d,2]+sqrt(sum((M1_expected.-M_actual).^2))
            vals[d,3]=vals[d,3]+sqrt(sum((M2_expected.-M_actual).^2))
            vals[d,4]=vals[d,4]+sqrt(sum((T1_expected.-T_actual).^2))
            vals[d,5]=vals[d,5]+sqrt(sum((T2_expected.-T_actual).^2))
        end
    end
    vals=vals./trials
    #println(vals)
    return vals
    #writedlm("SIRAccuracybySizeEta.csv",vals,',')
end

"""
    function benchmarkAllSIRDeterministic()

This function calculates time, memory, and prediction accuracy for the
deterministic SIR model methods.  This method compares these values across
N=[10,100,1000], Eps=[1x10^-3,1x10^-6,1x10^-9], and Change=[.01*θ,.1*θ,1*θ] for
θ in {η,δ}, averaged over the ten sets of parameters generated, and exports the
results to "SIRDeterministic.csv". Complex step 1x10^-6, dθ=.1*θ, and N=100 are
used when varying the other parameters.
"""
function benchmarkAllSIRDeterministic()
    params=readdlm("RealisticSIRParams.csv",',')
    trials=10
    direct=1
    complex=2
    iter=0
    vals=zeros(15,8)
    N=[10,100,1000]
    Eps=[1e-9,1e-6,1e-3]
    Change=[.01,.1,1]
    #set values for N loop
    change=.1
    eps=1e-6
    for n in N
        iter=iter+1
        vals[iter,1]=direct*trials
        vals[iter,2]=n*trials
        vals[iter,3]=eps*trials
        vals[iter,4]=change*trials
        for r in 1:trials
            delta=params[1,r]
            eta=params[2,r]
            ddelta=change*delta
            delta_2=delta+ddelta
            deta=change*eta
            eta_2=eta+deta
            test=median(@benchmark $SIRDirect($eta,$delta,$n))
            vals[iter,5]=vals[iter,5]+test.time
            vals[iter,6]=vals[iter,6]+test.memory
            derivs=SIRDirect(eta,delta,n)
            starting=derivs[1:3]
            eta_derivs=derivs[4:6]
            delta_derivs=derivs[7:9]
            expected_eta=starting.+(eta_derivs*deta)
            expected_delta=starting.+(delta_derivs*ddelta)
            actual_eta=SIRDirect(eta_2,delta,n)[1:3]
            actual_delta=SIRDirect(delta_2,eta,n)[1:3]
            vals[iter,7]=vals[iter,7]+sum(abs.(expected_eta.-actual_eta))/3
            vals[iter,8]=vals[iter,8]+sum(abs.(expected_delta.-actual_delta))/3
        end
        println(vals[iter,:]./trials)
        iter=iter+1
        vals[iter,1]=complex*trials
        vals[iter,2]=n*trials
        vals[iter,3]=eps*trials
        vals[iter,4]=change*trials
        for r in 1:trials
            delta=params[1,r]
            eta=params[2,r]
            ddelta=change*delta
            delta_2=delta+ddelta
            deta=change*eta
            eta_2=eta+deta
            test=median(@benchmark $SIRComplex($eta,$delta,$n))
            vals[iter,5]=vals[iter,5]+test.time
            vals[iter,6]=vals[iter,6]+test.memory
            derivs=SIRComplex(eta,delta,n)
            starting=derivs[1:3]
            eta_derivs=derivs[4:6]
            delta_derivs=derivs[7:9]
            expected_eta=starting.+(eta_derivs*deta)
            expected_delta=starting.+(delta_derivs*ddelta)
            actual_eta=SIRComplex(eta_2,delta,n)[1:3]
            actual_delta=SIRComplex(delta_2,eta,n)[1:3]
            vals[iter,7]=vals[iter,7]+sum(abs.(expected_eta.-actual_eta))/3
            vals[iter,8]=vals[iter,8]+sum(abs.(expected_delta.-actual_delta))/3
        end
        println(vals[iter,:]./trials)
    end
    #set values for change loop
    n=100
    eps=1e-6
    for change in Change
        iter=iter+1
        vals[iter,1]=direct*trials
        vals[iter,2]=n*trials
        vals[iter,3]=eps*trials
        vals[iter,4]=change*trials
        for r in 1:trials
            delta=params[1,r]
            eta=params[2,r]
            ddelta=change*delta
            delta_2=delta+ddelta
            deta=change*eta
            eta_2=eta+deta
            test=median(@benchmark $SIRDirect($eta,$delta,$n))
            vals[iter,5]=vals[iter,5]+test.time
            vals[iter,6]=vals[iter,6]+test.memory
            derivs=SIRDirect(eta,delta,n)
            starting=derivs[1:3]
            eta_derivs=derivs[4:6]
            delta_derivs=derivs[7:9]
            expected_eta=starting.+(eta_derivs*deta)
            expected_delta=starting.+(delta_derivs*ddelta)
            actual_eta=SIRDirect(eta_2,delta,n)[1:3]
            actual_delta=SIRDirect(delta_2,eta,n)[1:3]
            vals[iter,7]=vals[iter,7]+sum(abs.(expected_eta.-actual_eta))/3
            vals[iter,8]=vals[iter,8]+sum(abs.(expected_delta.-actual_delta))/3
        end
        println(vals[iter,:]./trials)
        iter=iter+1
        vals[iter,1]=complex*trials
        vals[iter,2]=n*trials
        vals[iter,3]=eps*trials
        vals[iter,4]=change*trials
        for r in 1:trials
            delta=params[1,r]
            eta=params[2,r]
            ddelta=change*delta
            delta_2=delta+ddelta
            deta=change*eta
            eta_2=eta+deta
            test=median(@benchmark $SIRComplex($eta,$delta,$n))
            vals[iter,5]=vals[iter,5]+test.time
            vals[iter,6]=vals[iter,6]+test.memory
            derivs=SIRComplex(eta,delta,n)
            starting=derivs[1:3]
            eta_derivs=derivs[4:6]
            delta_derivs=derivs[7:9]
            expected_eta=starting.+(eta_derivs*deta)
            expected_delta=starting.+(delta_derivs*ddelta)
            actual_eta=SIRComplex(eta_2,delta,n)[1:3]
            actual_delta=SIRComplex(delta_2,eta,n)[1:3]
            vals[iter,7]=vals[iter,7]+sum(abs.(expected_eta.-actual_eta))/3
            vals[iter,8]=vals[iter,8]+sum(abs.(expected_delta.-actual_delta))/3
        end
        println(vals[iter,:]./trials)
    end
    #set values for eps loop
    n=100
    change=.1
    for eps in Eps
        iter=iter+1
        vals[iter,1]=complex*trials
        vals[iter,2]=n*trials
        vals[iter,3]=eps*trials
        vals[iter,4]=change*trials
        for r in 1:trials
            delta=params[1,r]
            eta=params[2,r]
            ddelta=change*delta
            delta_2=delta+ddelta
            deta=change*eta
            eta_2=eta+deta
            test=median(@benchmark $SIRComplex($eta,$delta,$n))
            vals[iter,5]=vals[iter,5]+test.time
            vals[iter,6]=vals[iter,6]+test.memory
            derivs=SIRComplex(eta,delta,n)
            starting=derivs[1:3]
            eta_derivs=derivs[4:6]
            delta_derivs=derivs[7:9]
            expected_eta=starting.+(eta_derivs*deta)
            expected_delta=starting.+(delta_derivs*ddelta)
            actual_eta=SIRComplex(eta_2,delta,n)[1:3]
            actual_delta=SIRComplex(delta_2,eta,n)[1:3]
            vals[iter,7]=vals[iter,7]+sum(abs.(expected_eta.-actual_eta))/3
            vals[iter,8]=vals[iter,8]+sum(abs.(expected_delta.-actual_delta))/3
        end
        println(vals[iter,:]./trials)
    end
    vals=vals./trials
    println(vals)
    writedlm("SIRDeterministic.csv",vals,',')
end

#=
Standardized parameters for Branching Process

This section includes additional methods for the Benchmarking code that allow
for the standardization of parameters across methods.
=#

"""
    function cumulative_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20)::Array

This function is identical to cumulative_inverse with standardized parameters.
The threshold parameter is not used.

# Example
```julia-repl
julia> cumulative_inverse([1;2],[2;3],[-1 1;1 -1])
2×2 Array{Float64,2}:
 2.66667  2.0
 1.33333  4.0
```
"""
function cumulative_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20)::Array
    return cumulative_inverse(calculate_offspring(birth,death,migration))
end

"""
    function cumulative_iterative(birth::Array,death::Array,migration::Array,threshold::Number=1e-20)::Array

This function is identical to cumulative_iterative with standardized parameters.

# Example
```julia-repl
julia> cumulative_iterative([1;2],[2;3],[-1 1;1 -1])
2×2 Array{Float64,2}:
 2.66667  2.0
 1.33333  4.0
```
"""
function cumulative_iterative(birth::Array,death::Array,migration::Array,threshold::Number=1e-20)::Array
    cumulative_iterative(calculate_offspring(birth,death,migration),threshold)
end

"""
    function dAddelta_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dAdtheta_inverse with standardized parameters.
The parameters threshold and eps are not used.  A is calcuated using
cumulative_inverse.  The derivative is taken with respect to δ[1].

# Example
```julia-repl
julia> dAddelta_inverse([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 -1.11111   -1.33333
 -0.555556  -0.666667
```
"""
function dAddelta_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    return dAdtheta_inverse(cumulative_inverse(birth,death,migration),dF_ddelta(birth,death,migration,1))
end

"""
    function dAdbeta_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dAdtheta_inverse with standardized parameters.
The parameters threshold and eps are not used.  A is calcuated using
cumulative_inverse.  The derivative is taken with respect to β[1].

# Example
```julia-repl
julia> dAdbeta_inverse([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 2.44444  1.33333
 1.22222  0.666667
```
"""
function dAdbeta_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    return dAdtheta_inverse(cumulative_inverse(birth,death,migration),dF_dbeta(birth,death,migration,1))
end
"""
    function dAdlambda_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dAdtheta_inverse with standardized parameters.
The parameters threshold and eps are not used.  A is calcuated using
cumulative_inverse.  The derivative is taken with respect to λ[1,2].

# Example
```julia-repl
julia> dAdlambda_inverse([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 -0.222222  1.33333
 -0.111111  0.666667
```
"""
function dAdlambda_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    return dAdtheta_inverse(cumulative_inverse(birth,death,migration),dF_dlambda(birth,death,migration,1,2))
end

"""
    function dAddelta_iterative(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dAddelta_iterative with standardized parameters.
The eps parameter is not used.  A is calculated using cumulative_iterative, and
the derivative is taken with respect to δ[1].

# Example
```julia-repl
julia> dAddelta_iterative([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 -1.11111   -1.33333
 -0.555556  -0.666667
```
"""
function dAddelta_iterative(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    return dAddelta_iterative(cumulative_iterative(birth,death,migration,threshold),birth,death,migration,1,threshold)
end

"""
    function dAdbeta_iterative(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dAdbeta_iterative with standardized parameters.
The eps parameter is not used.  A is calculated using cumulative_iterative, and
the derivative is taken with respect to β[1].

# Example
```julia-repl
julia> dAdbeta_iterative([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 2.44444  1.33333
 1.22222  0.666667
```
"""
function dAdbeta_iterative(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    return dAdbeta_iterative(cumulative_iterative(birth,death,migration,threshold),birth,death,migration,1,threshold)
end

"""
    function dAdlambda_iterative(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dAdlambda_iterative with standardized parameters.
The eps parameter is not used.  A is calculated using cumulative_iterative, and
the derivative is taken with respect to λ[1,2].

# Example
```julia-repl
julia> dAdlambda_iterative([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 -0.222222  1.33333
 -0.111111  0.666667
```
"""
function dAdlambda_iterative(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    return dAdlambda_iterative(cumulative_iterative(birth,death,migration,threshold),birth,death,migration,1,2,threshold)
end

"""
    function dAddelta_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dAddelta_complex with standardized parameters.
The derivative is taken with respect to δ[1].

# Example
```julia-repl
julia> dAddelta_complex([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 -1.11111   -1.33333
 -0.555556  -0.666667
```
"""
function dAddelta_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    step=convert(typeof(birth[1]),eps)
    return dAddelta_complex(birth,death,migration,1,step,threshold)
end

"""
    function dAdbeta_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dAdbeta_complex with standardized parameters.
The derivative is taken with respect to β[1].

# Example
```julia-repl
julia> dAdbeta_complex([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 2.44444  1.33333
 1.22222  0.666667
```
"""
function dAdbeta_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    step=convert(typeof(birth[1]),eps)
    return dAdbeta_complex(birth,death,migration,1,step,threshold)
end

function dAdbeta_complexinverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    step=convert(typeof(birth[1]),eps)
    return dAdbeta_complexinverse(birth,death,migration,1,step)
end

"""
    function dAdlambda_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dAdlambda_complex with standardized parameters.
The derivative is taken with respect to λ[1,2].

# Example
```julia-repl
julia> dAdlambda_complex([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 -0.222222  1.33333
 -0.111111  0.666667
```
"""
function dAdlambda_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    step=convert(typeof(birth[1]),eps)
    return dAdlambda_complex(birth,death,migration,1,2,step,threshold)
end

"""
    function deddelta_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dedtheta_inverse with standardized parameters.
The eps and threshold parameters are not used.  The derivative is taken with
respect to δ[1].

# Example
```julia-repl
julia> deddelta_inverse([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2-element Array{Float64,1}:
 2.5905203907920286e-16
 1.2952601953960123e-16
```
"""
function deddelta_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    e=extinction_probability(birth,death,migration,threshold)
    return dedtheta_inverse(birth,death,migration,e,dPddelta_inverse(birth,death,migration,1,e))
end

"""
    function dedbeta_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dedtheta_inverse with standardized parameters.
The eps and threshold parameters are not used.  The derivative is taken with
respect to β[1].

# Example
```julia-repl
julia> dedbeta_inverse([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2-element Array{Float64,1}:
 -3.3306690738754657e-16
 -1.6653345369377304e-16
```
"""
function dedbeta_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    e=extinction_probability(birth,death,migration,threshold)
    return dedtheta_inverse(birth,death,migration,e,dPdbeta_inverse(birth,death,migration,1,e))
end

"""
    function dedlambda_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dedtheta_inverse with standardized parameters.
The eps and threshold parameters are not used.  The derivative is taken with
respect to λ[1,2].

# Example
```julia-repl
julia> dedlambda_inverse([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
julia> dedlambda_inverse([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2-element Array{Float64,1}:
 -1.4802973661668736e-16
 -7.401486830834357e-17
```
"""
function dedlambda_inverse(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    e=extinction_probability(birth,death,migration,threshold)
    return dedtheta_inverse(birth,death,migration,e,dPdlambda_inverse(birth,death,migration,1,2,e))
end

"""
    function de_ddelta(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to de_ddelta with standardized parameters.  The eps
parameter is not used.  The derivative is taken with respect to δ[1].

# Example
```julia-repl
julia> de_ddelta([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2-element Array{Float64,1}:
 2.960411911137605e-16
 1.4800751926283627e-16
```
"""
function de_ddelta(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    return de_ddelta(extinction_probability(birth,death,migration,threshold),birth,death,migration,1,threshold)
end

"""
    function de_dbeta(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to de_dbeta with standardized parameters.  The eps
parameter is not used.  The derivative is taken with respect to β[1].

# Example
```julia-repl
julia> de_dbeta([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2-element Array{Float64,1}:
 -3.330503750644931e-16
 -1.6651336278095725e-16
```
"""
function de_dbeta(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    return de_dbeta(extinction_probability(birth,death,migration,threshold),birth,death,migration,1,threshold)
end

"""
    function de_dlambda(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to de_dlambda with standardized parameters.  The eps
parameter is not used.  The derivative is taken with respect to λ[1,2]}.

# Example
```julia-repl
julia> de_dlambda([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2-element Array{Float64,1}:
 -1.4801213582890356e-16
 -7.399347894329741e-17
```
"""
function de_dlambda(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    return de_dlambda(extinction_probability(birth,death,migration,threshold),birth,death,migration,1,2,threshold)
end

"""
    function deddelta_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to deddelta_complex with standardized parameters.
The derivative is taken with respect to δ[1].

# Example
```julia-repl
julia> deddelta_complex([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2-element Array{Float64,1}:
 2.1705219273391446e-15
 2.5642927199289465e-15
```
"""
function deddelta_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    step=convert(typeof(birth[1]),eps)
    return deddelta_complex(birth,death,migration,1,step,threshold)
end

"""
    function dedbeta_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dedeta_complex with standardized parameters.
The derivative is taken with respect to β[1].

# Example
```julia-repl
julia> dedbeta_complex([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2-element Array{Float64,1}:
 -3.1234339930002325e-15
 -3.769978218160122e-15
```
"""
function dedbeta_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    step=convert(typeof(birth[1]),eps)
    return dedbeta_complex(birth,death,migration,1,step,threshold)
end

"""
    function dedlambda_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dedlambda_complex with standardized parameters.
The derivative is taken with respect to λ[1,2].

# Example
```julia-repl
julia> dedlambda_complex([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
2-element Array{Float64,1}:
 -1.138200522872966e-15
 -1.3357325326677992e-15
```
"""
function dedlambda_complex(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array
    step=convert(typeof(birth[1]),eps)
    return dedlambda_complex(birth,death,migration,1,2,step,threshold)
end

"""
    function drho_ddelta(birth::Array,death::Array,migration::Array)::Array

This function is identical to drho_dtheta with standardized parameters. The
derivative is taken with respect to δ[1].

# Example
```julia-repl
julia> drho_ddelta([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
1-element Array{Float64,1}:
 -0.5000000000000001
```
"""
function drho_ddelta(birth::Array,death::Array,migration::Array)::Array
    return drho_dtheta(calculate_generator(birth,death,migration),convert(Array{typeof(birth[1]),2},domega_ddelta(length(birth),1)))
end

"""
    function drho_dbeta(birth::Array,death::Array,migration::Array)::Array

This function is identical to drho_dtheta with standardized parameters. The
derivative is taken with respect to β[1].

# Example
```julia-repl
julia> drho_dbeta([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
1-element Array{Float64,1}:
 0.5000000000000001
```
"""
function drho_dbeta(birth::Array,death::Array,migration::Array)::Array
    return drho_dtheta(calculate_generator(birth,death,migration),convert(Array{typeof(birth[1]),2},domega_dbeta(length(birth),1)))
end

"""
    function drho_dlambda(birth::Array,death::Array,migration::Array)::Array

This function is identical to drho_dtheta with standardized parameters.  The
derivative is taken with respect to λ[1,2].

# Example
```julia-repl
julia> drho_dlambda([1.0;2.0],[2.0;3.0],[-1.0 1.0;1.0 -1.0])
1-element Array{Float64,1}:
 -1.0000000000000002
```
"""
function drho_dlambda(birth::Array,death::Array,migration::Array)::Array
    return drho_dtheta(calculate_generator(birth,death,migration),convert(Array{typeof(birth[1]),2},domega_dlambda(length(birth),1,2)))
end

"""
    function dmean_dbeta(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dmean_dtheta with standardized parameters.
The derivative is taken with respect to β[1] at time t=2.  The step parameter
is not used.

# Example
```julia-repl
julia> dmean_dbeta([1.0;2.0],[2.0;1.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 1.38934  -0.203417
 2.79707  -0.793459
```
"""
function dmean_dbeta(birth::Array,death::Array,migration::Array,threshold::Number=1e-16,eps::Number=1e-6)::Array
    n=size(birth)[1]
    return dmean_dtheta(calculate_generator(birth,death,migration),domega_dbeta(n,1),2,threshold)
end

"""
    function dmean_ddelta(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dmean_dtheta with standardized parameters.
The derivative is taken with respect to δ[1] at time t=2.  The step parameter is
not used.

# Example
```julia-repl
julia> dmean_ddelta([1.0;2.0],[2.0;1.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 -1.38934  0.203417
 -2.79707  0.793459
```
"""
function dmean_ddelta(birth::Array,death::Array,migration::Array,threshold::Number=1e-16,eps::Number=1e-6)::Array
    n=size(birth)[1]
    return dmean_dtheta(calculate_generator(birth,death,migration),domega_ddelta(n,1),2,threshold)
end

"""
    function dmean_dlambda(birth::Array,death::Array,migration::Array,threshold::Number=1e-20,eps::Number=1e-6)::Array

This function is identical to dmean_dtheta with standardized parameters.
The derivative is taken with respect to λ[1,2] at time t=2.  The step parameter
is not used.

# Example
```julia-repl
julia> dmean_dlambda([1.0;2.0],[2.0;1.0],[-1.0 1.0;1.0 -1.0])
2×2 Array{Float64,2}:
 -1.59275  1.18592
 -3.59053  2.00361
```
"""
function dmean_dlambda(birth::Array,death::Array,migration::Array,threshold::Number=1e-16,eps::Number=1e-6)::Array
    n=size(birth)[1]
    return dmean_dtheta(calculate_generator(birth,death,migration),domega_dlambda(n,1,2),2,threshold)
end

#=
Standardized parameters for SIR model

This section includes additional methods for the Benchmarking code that allow
for the standardization of parameters across methods.

=#

"""
    function dm_ddelta(N::Integer,delta::Number,eta::Number,eps::Number,TP::Type=Float64)::Array

This function is identical to dm_ddelta with standardized parameters.

# Example
```julia-repl
julia> dm_ddelta(3,1,1,1e-6)
4-element Array{Float64,1}:
  0.0
 -0.4575000000000001
 -0.28125
  0.0
```
"""
function dm_ddelta(N::Integer,delta::Number,eta::Number,eps::Number,TP::Type=Float64)::Array
    return dm_ddelta(N,delta,eta,TP)
end

"""
    function dt_ddelta(N::Integer,delta::Number,eta::Number,eps::Number,TP::Type=Float64)::Array

This function is identical to dt_ddelta with standardized parameters.

# Example
```julia-repl
julia> dt_ddelta(3,1,1,1e-6)
4-element Array{Float64,1}:
  0.0
 -1.4770833333333335
 -1.786458333333333
 -1.8333333333333333
```
"""
function dt_ddelta(N::Integer,delta::Number,eta::Number,eps::Number,TP::Type=Float64)::Array
    return dt_ddelta(N,delta,eta,TP)
end

"""
    function dm_deta(N::Integer,delta::Number,eta::Number,eps::Number,TP::Type=Float64)::Array

This function is identical to dm_deta with standardized parameters.

# Example
```julia-repl
julia> dm_deta(3,1,1,1e-6)
4-element Array{Float64,1}:
 0.0
 0.4575000000000001
 0.28125
 0.0
```
"""
function dm_deta(N::Integer,delta::Number,eta::Number,eps::Number,TP::Type=Float64)::Array
    return dm_deta(N,delta,eta,TP)
end

"""
    function dt_deta(N::Integer,delta::Number,eta::Number,eps::Number,TP::Type=Float64)::Array

This function is identical to dt_deta with standardized parameters.

# Example
```julia-repl
julia> dt_deta(3,1,1,1e-6)
4-element Array{Float64,1}:
 0.0
 0.2062500000000001
 0.109375
 0.0
```
"""
function dt_deta(N::Integer,delta::Number,eta::Number,eps::Number,TP::Type=Float64)::Array
    return dt_deta(N,delta,eta,TP)
end

"""
    function SIRDirect(N::Number,delta::Number,eta::Number)::Array

This function is identical to solveSystem with SIRDerivs!, tspan=(0,20), and
standardized parameters.

# Example
```julia-repl
julia> SIRDirect(10.0,.5,.5)
9-element Array{Float64,1}:
   6.153822753491371
   0.044663297351630375
   3.8015139491570005
 -11.757932373451958
   0.25423339264319894
  11.503698980808757
  11.208294553987672
  -0.5977879627825006
 -10.610506591205173
```
"""
function SIRDirect(N::Number,delta::Number,eta::Number)::Array
    return solveSystem(SIRDerivs!,[N-1,1.0,0,0,0,0,0,0,0],[eta,delta,N],(0.0,20.0))
end

"""
    function SIRComplex(N::Number,delta::Number,eta::Number,eps::Number=1e-6)::Array

This function is identical to solveSystem with SIR!, tspan=(0,20), and
solveSystem_complex with parameters 1 and 2 all concatenated into a single
array, with standardized parameters.

# Example
```julia-repl
julia> julia> SIRComplex(10.0,.5,.5)
9-element Array{Float64,1}:
   6.1538364317412
   0.044663488723492926
   3.801500079535306
 -11.757940877217914
   0.254237148812786
  11.50370372840513
  11.208246937506885
  -0.5977999300270691
 -10.610447007479815
```
"""
function SIRComplex(N::Number,delta::Number,eta::Number,eps::Number=1e-6)::Array
    u0=[N-1,1.0,0]
    p=[eta,delta,N]
    tspan=(0.0,20.0)
    return [solveSystem(SIR!,u0,p,tspan);solveSystem_complex(SIR!,u0,p,tspan,1,eps);solveSystem_complex(SIR!,u0,p,tspan,2,eps)]
end
