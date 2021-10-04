#=
Code for Birth Notes on Birth-Death Migration Processes (Kenneth Lange)
code by Rachel Mester
last updated October 21 2020
=#

using LinearAlgebra
using DifferentialEquations
using QuadGK

#=
Branching Process

This section includes functions to compute the following quantities for
multitype branching processes using the following methods:
    A (expected cumulative number of particles)
        inverse method
        iterative method
    e (extinction probability)
        iterative method
    dA/dθ (where θ in {β[i],δ[i],λ[i,j]})
        inverse method
        iterative method
        complex step method
    de/dθ (where θ in {β[i],δ[i],λ[i,j]})
        iterative method
        inverse
        complex step method
    dρ/dθ (where ρ is the dominant eigenvalue and θ in {β[i],δ[i],λ[i,j]})
        eigenvector method
    dM(t)/dθ (where M(t) is the expected number of each type at time t and θ
    in {β[i],δ[i],λ[i,j]})
        integration method
=#

"""
    cumulative_inverse(offspring::Array)::Array

Using the inverse method, return the matrix (A) with entries a_{ij} of the expected number of
particles of type j ultimately generated starting with 1 particle of type i.

Use the offspring matrix (F) calculated in the function `calculate_offspring`.

# Example
```julia-repl
julia> cumulative_inverse([1/3 1/2;1/6 2/3])
2×2 Array{Float64,2}:
 2.4  3.6
 1.2  4.8
```
"""
function cumulative_inverse(offspring::Array)::Array
    return inv(I-offspring)
end

"""
    cumulative_iterative(offspring::Array,threshold::Number=1e-20)::Array

Using the iteration method, return the matrix (A) with entries a_{ij} of the expected number of
particles of type j ultimately generated starting with 1 particle of type i.

Use the offspring matrix (F) calculated in the function `calculate_offspring`.

# Example
```julia-repl
julia> cumulative_iterative([1/3 1/2;1/6 2/3])
2×2 Array{Float64,2}:
 2.4  3.6
 1.2  4.8
```
"""
function cumulative_iterative(offspring::Array,threshold::Number=1e-20)::Array
    iter=0
    n=size(offspring,1)
    A=zeros(typeof(offspring[1,1]),n,n)
    previous_iterate=similar(A)
    distance=Inf
    while distance > threshold
        previous_iterate=copy(A)
        A=I+offspring*A
        mul!(A,offspring,previous_iterate)
        A.=A+I
        distance=abs(sum(A)-sum(previous_iterate))
        iter=iter+1
    end
    return A
end

"""
    extinction_probability(birth::Array,death::Array,migration::Array,threshold::Number=1e-20)::Array
Return the array (e) with entries e_i of the probability that the process will
go extinct starting with one particle of type i.

Use the migration generator (Γ) calculated in the function
`migration_generator`.

# Example
```julia-repl
julia> extinction_probability([1;2],[2;3],[-3 3;1 -1])
2-element Array{Float64,1}:
 0.9999999999999994
 0.9999999999999993
```
"""
function extinction_probability(birth::Array,death::Array,migration::Array,threshold::Number=1e-20)::Array
    n=length(birth)
    iters=0
    extinction=zeros(typeof(birth[1]),n)
    previous_iterate=similar(extinction)
    total_risk=birth+death-diag(migration)
    coefficients=[death./total_risk birth./total_risk (migration-Diagonal(migration))./total_risk]
    distance=Inf
    while distance>threshold
        previous_iterate=copy(extinction)
        extinction=coefficients[:,1]+coefficients[:,2].*previous_iterate.*previous_iterate+coefficients[:,3:end]*previous_iterate
        distance=abs(sum(extinction-previous_iterate))
        iters=iters+1
    end
    return extinction
end

"""
     dAdtheta_inverse(A::Array,dF_dtheta::Array)::Array

 Using the inverse method, return the derivative of A (cumulative number of particles) with respect to
 θ (one of the rate parameters).

 Use the matrix A calculated in the function `cumulative_inverse` and the
 derivative matrix (dF_dtheta) calculated in one of the following functions:
  `dF_dbeta`, `dF_ddelta`, `dF_dlambda`.

 # Example
 ```julia-repl
 julia> dAdtheta_inverse([2.4 3.6;1.2 4.8],[.278 -.0833;0 0])
 2×2 Array{Float64,2}:
  1.36138   1.4423
  0.680688  0.721152
   ```
 """
function dAdtheta_inverse(A::Array,dF_dtheta::Array)::Array
    return A*dF_dtheta*A
end

"""
    function dAdbeta_iterative(A::Array,birth::Array,death::Array,migration::Array,i::Integer,threshold::Number=1e-20)::Array

Using the iterative method, return the derivative of A (cumulative number of particles) with respect to
β[i] (the rate of birth of type i).

Use the matrix A calculated in the function `cumulative_inverse` and the
migration generator (Γ) calculated from `migration_generator`.

# Example
```julia-repl
julia> dAdbeta_iterative([2.4 3.6;1.2 4.8],[1;2],[2;3],[-3 3;1 -1],1)
2×2 Array{Float64,2}:
 1.36  1.44
 0.68  0.72
  ```
"""
function dAdbeta_iterative(A::Array,birth::Array,death::Array,migration::Array,i::Integer,threshold::Number=1e-20)::Array
    n=length(birth)
    iter=0
    previous_iterate=similar(A)
    dA_dbeta=zeros(typeof(A[1,1]),n,n)
    distance=Inf
    total_risk=birth+death-diag(migration)
    totalrisk_squared=total_risk.*total_risk
    lambda=migration-Diagonal(migration)
    coefficients=[2*birth./total_risk lambda./total_risk]
    i_coefficient=[(transpose(2(death[i]-migration[i,i])*A[i,:])+transpose(-lambda[i,:])*A)/totalrisk_squared[i] 2*birth[i]/total_risk[i] transpose(lambda[i,:]/total_risk[i])]
    while distance>threshold
        previous_iterate=copy(dA_dbeta)
        dA_dbeta=coefficients[:,1].*previous_iterate+coefficients[:,2:end]*previous_iterate
        dA_dbeta[i,:]=vec(i_coefficient[:,1:n])+i_coefficient[n+1]*previous_iterate[i,:]+vec(i_coefficient[:,n+2:end]*previous_iterate)
        distance=abs(sum(dA_dbeta-previous_iterate))
        iter=iter+1
    end
    return dA_dbeta
end

"""
    function dAddelta_iterative(A::Array,birth::Array,death::Array,migration::Array,i::Integer,threshold::Number=1e-20)::Array

Using the iterative method, return the derivative of A (cumulative number of
particles) with respect to δ[i] (the rate of death of type i).

Use the matrix A calculated in the function `cumulative_inverse` and the
migration generator (Γ) calculated from `migration_generator`.

# Example
```julia-repl
julia> dAddelta_iterative([2.4 3.6;1.2 4.8],[1;2],[2;3],[-3 3;1 -1],1)
2×2 Array{Float64,2}:
 -0.56  -1.44
 -0.28  -0.72
  ```
"""
function dAddelta_iterative(A::Array,birth::Array,death::Array,migration::Array,i::Integer,threshold::Number=1e-20)::Array
    n=length(birth)
    iter=0
    previous_iterate=similar(A)
    dA_ddelta=zeros(typeof(A[1,1]),n,n)
    distance=Inf
    total_risk=birth+death-diag(migration)
    totalrisk_squared=total_risk.*total_risk
    lambda=migration-Diagonal(migration)
    coefficients=[2*birth./total_risk lambda./total_risk]
    i_coefficient=[(transpose(-2birth[i]*A[i,:])+transpose(-lambda[i,:])*A)./totalrisk_squared[i] 2*birth[i]/total_risk[i] transpose(lambda[i,:]/total_risk[i])]
    while distance>threshold
        previous_iterate=copy(dA_ddelta)
        dA_ddelta=coefficients[:,1].*previous_iterate+coefficients[:,2:end]*previous_iterate
        dA_ddelta[i,:]=vec(i_coefficient[:,1:n])+i_coefficient[n+1]*previous_iterate[i,:]+vec(i_coefficient[:,n+2:end]*previous_iterate)
        distance=abs(sum(dA_ddelta-previous_iterate))
        iter=iter+1
    end
    return dA_ddelta
end

"""
    function dAdlambda_iterative(A::Array,birth::Array,death::Array,migration::Array,i::Integer,j::Integer,threshold::Number=1e-20)::Array

Using the iterative method, return the derivative of A (cumulative number of particles) with respect to
λ[i,j] (the rate of migration from type i to type j where i is not equal to j).

Use the matrix A calculated in the function `cumulative_inverse` and the
migration generator (Γ) calculated from `migration_generator`.

# Example
```julia-repl
julia> dAdlambda_iterative([2.4 3.6;1.2 4.8],[1;2],[2;3],[-3 3;1 -1],1,2)
2×2 Array{Float64,2}:
 -0.08  0.48
 -0.04  0.24
  ```
"""
function dAdlambda_iterative(A::Array,birth::Array,death::Array,migration::Array,i::Integer,j::Integer,threshold::Number=1e-20)::Array
    n=length(birth)
    iter=0
    previous_iterate=similar(A)
    dA_dlambda=zeros(typeof(A[1,1]),n,n)
    distance=Inf
    total_risk=birth+death-diag(migration)
    totalrisk_squared=total_risk.*total_risk
    lambda=migration-Diagonal(migration)
    coefficients=[2*birth./total_risk lambda./total_risk]
    i_coefficient=[(transpose(-2*birth[i]*A[i,:])-transpose(lambda[i,:])*A)/totalrisk_squared[i]+transpose(A[j,:])/total_risk[i] 2*birth[i]/total_risk[i] transpose(lambda[i,:]/total_risk[i])]
    while distance>threshold
        previous_iterate=copy(dA_dlambda)
        dA_dlambda=coefficients[:,1].*previous_iterate+coefficients[:,2:end]*previous_iterate
        dA_dlambda[i,:]=vec(i_coefficient[:,1:n])+i_coefficient[n+1]*previous_iterate[i,:]+vec(i_coefficient[:,n+2:end]*previous_iterate)
        distance=abs(sum(dA_dlambda-previous_iterate))
        iter=iter+1
    end
    return dA_dlambda
end

"""
    function dAdbeta_complex(birth::Array,death::Array,migration::Array,i::Integer,eps::Number=1e-6)::Array

Using the iterative complex step method, return the derivative of A (cumulative
number of particles) with respect to β[i] (the rate of birth of type i).

Use the migration generator (Γ) calculated from `migration_generator`.
In order to complete computations in single precision, change the 'eps'
parameter to 1f-6 (or similar).

# Example
```julia-repl
julia> dAdbeta_complex([1.0;2],[2;3],[-3 3;1 -1],1)
2×2 Array{Float64,2}:
 1.36  1.44
 0.68  0.72
  ```
"""
function dAdbeta_complex(birth::Array,death::Array,migration::Array,i::Integer,eps::Number=1e-6,threshold::Number=1e-20)::Array
    birth=complex(birth)
    birth[i]=birth[i]+eps*im
    offspring=calculate_offspring(birth,death,migration)
    A=cumulative_iterative(offspring,threshold)
    return imag(A)./eps
end

function dAdbeta_complexinverse(birth::Array, death::Array, migration::Array, i::Integer, eps::Number=1e-6)::Array
    birth=complex(birth)
    birth[i]=birth[i]+eps*im
    offspring=calculate_offspring(birth,death,migration)
    A=cumulative_inverse(offspring)
    return imag(A)./eps
end
    
"""
    function dAddelta_complex(birth::Array,death::Array,migration::Array,i::Integer,eps::Number=1e-6)::Array

Using the iterative complex step method, return the derivative of A (cumulative
number of particles) with respect to δ[i] (the rate of death of type i).

Use the migration generator (Γ) calculated from `migration_generator`.
In order to complete computations in single precision, change the 'eps'
parameter to 1f-6 (or similar).

# Example
```julia-repl
julia> dAddelta_complex([1;2],[2.0;3],[-3 3;1 -1],1)
2×2 Array{Float64,2}:
 -0.56  -1.44
 -0.28  -0.72
  ```
"""
function dAddelta_complex(birth::Array,death::Array,migration::Array,i::Integer,eps::Number=1e-6,threshold::Number=1e-20)::Array
    death=complex(death)
    death[i]=death[i]+eps*im
    offspring=calculate_offspring(birth,death,migration)
    A=cumulative_iterative(offspring,threshold)
    return imag(A)./eps
end

"""
    function dAdlambda_complex(birth::Array,death::Array,migration::Array,i::Integer,j::Integer,eps::Number=1e-6)::Array

Using the iterative complex step method, return the derivative of A (cumulative
number of particles) with respect to λ[i,j] (the rate of migration from
type i to type j where i is not equal to j).
In order to complete computations in single precision, change the 'eps'
parameter to 1f-6 (or similar).

# Example
```julia-repl
julia> dAdlambda_complex([1;2],[2;3],[-3 3.0;1 -1],1,2)
2×2 Array{Float64,2}:
 -0.08  0.48
 -0.04  0.24
  ```
"""
function dAdlambda_complex(birth::Array,death::Array,migration::Array,i::Integer,j::Integer,eps::Number=1e-6,threshold::Number=1e-20)::Array
    migration=complex(migration)
    migration[i,j]=migration[i,j]+eps*im
    migration=migration_generator(migration)
    offspring=calculate_offspring(birth,death,migration)
    A=cumulative_iterative(offspring,threshold)
    return imag(A)./eps
end

"""
    de_dbeta(e::Array,birth::Array,death::Array,migration::Array,i::Integer,threshold::Number=1e-20)::Array

Return the derivative of e (the extinction probabilities) with respect to
β[i] (the birth rate of type i).

Use the extinction probabilities (e) from `extinction_probability` and the
migration generator (Γ) calculated in the function `migration_generator`.

# Example
```julia-repl
julia> de_dbeta(extinction_probability([1;2],[2;1],[-1 1;1 -1]),[1;2],[2;1],[-1 1;1 -1],1)
2-element Array{Float64,1}:
 -0.0851978153117322
 -0.08997251113059712
```
"""
function de_dbeta(e::Array,birth::Array,death::Array,migration::Array,i::Integer,threshold::Number=1e-20)::Array
    n=length(birth)
    iters=0
    previous_iterate=similar(e)
    de_dbeta=zeros(typeof(e[1]),n)
    distance=Inf
    total_risk=birth+death-diag(migration)
    totalrisk_squared=total_risk.*total_risk
    lambda=migration-Diagonal(migration)
    coefficients=[2*birth.*e./total_risk lambda./total_risk]
    i_coefficient=[((death[i]-migration[i,i])*e[i]^2-death[i]-transpose(lambda[i,:])*e)/totalrisk_squared[i] 2*birth[i]*e[i]/total_risk[i] transpose(lambda[i,:]/total_risk[i])]
    while distance>threshold
        previous_iterate=copy(de_dbeta)
        de_dbeta=coefficients[:,1].*previous_iterate+coefficients[:,2:end]*previous_iterate
        de_dbeta[i]=i_coefficient[1]+i_coefficient[2]*previous_iterate[i]+i_coefficient[:,3:end]*previous_iterate
        distance=abs(sum(de_dbeta-previous_iterate))
        iters=iters+1
    end
    return de_dbeta
end

"""
    de_ddelta(e::Array,birth::Array,death::Array,migration::Array,i::Integer,threshold::Number=1e-20)::Array

Return the derivative of e (the extinction probabilities) with respect to
δ[i] (the death rate of type i).

Use the extinction probabilities (e) from `extinction_probability` and the
migration generator (Γ) calculated in the function `migration_generator`.

# Example
```julia-repl
julia> de_ddelta(extinction_probability([1;2],[2;1],[-1 1;1 -1]),[1;2],[2;1],[-1 1;1 -1],1)
2-element Array{Float64,1}:
 0.09595266340475808
 0.1013300874512707
```
"""
function de_ddelta(e::Array,birth::Array,death::Array,migration::Array,i::Integer,threshold::Number=1e-20)::Array
    iters=0
    n=length(birth)
    previous_iterate=similar(e)
    de_ddelta=zeros(typeof(e[1]),n)
    distance=Inf
    total_risk=birth+death-diag(migration)
    totalrisk_squared=total_risk.*total_risk
    lambda=migration-Diagonal(migration)
    coefficients=[2*birth.*e./total_risk lambda./total_risk]
    i_coefficient=[(1/total_risk[i])-(death[i]+birth[i]*e[i]^2+transpose(lambda[i,:])*e)/totalrisk_squared[i] 2*birth[i]*e[i]/total_risk[i] transpose(lambda[i,:]/total_risk[i])]
    while distance>threshold
        previous_iterate=copy(de_ddelta)
        de_ddelta=coefficients[:,1].*previous_iterate+coefficients[:,2:end]*previous_iterate
        de_ddelta[i]=i_coefficient[1]+i_coefficient[2]*previous_iterate[i]+i_coefficient[:,3:end]*previous_iterate
        distance=abs(sum(de_ddelta-previous_iterate))
        iters=iters+1
    end
    return de_ddelta
end

"""
    de_dlambda(e::Array,birth::Array,death::Array,migration::Array,i::Integer,j::Integer,threshold::Number=1e-20)::Array

Return the derivative of e (the extinction probabilities) with respect to
λ[i,j] (the migration rate from type i to type j where i is not equal
to j).

Use the extinction probabilities (e) from `extinction_probability` and the
migration generator (Γ) calculated in the function `migration_generator`.

# Example
```julia-repl
julia> de_dlambda(extinction_probability([1;2],[2;1],[-1 1;1 -1]),[1;2],[2;1],[-1 1;1 -1],1,2)
2-element Array{Float64,1}:
 -0.10670751149778374
 -0.11268766377194409
```
"""
function de_dlambda(e::Array,birth::Array,death::Array,migration::Array,i::Integer,j::Integer,threshold::Number=1e-20)::Array
    n=length(birth)
    iters=0
    previous_iterate=similar(e)
    de_dlambda=zeros(typeof(e[1]),n)
    distance=Inf
    total_risk=birth+death-diag(migration)
    totalrisk_squared=total_risk.*total_risk
    lambda=migration-Diagonal(migration)
    coefficients=[2*birth.*e./total_risk lambda./total_risk]
    i_coefficient=[(e[j]/total_risk[i])-(birth[i]*e[i]^2+death[i]+transpose(lambda[i,:])*e)/totalrisk_squared[i] 2*birth[i]*e[i]/total_risk[i] transpose(lambda[i,:]/total_risk[i])]
    while distance>threshold
        previous_iterate=copy(de_dlambda)
        de_dlambda=coefficients[:,1].*previous_iterate+coefficients[:,2:end]*previous_iterate
        de_dlambda[i]=i_coefficient[1]+i_coefficient[2]*previous_iterate[i]+i_coefficient[:,3:end]*previous_iterate
        distance=abs(sum(de_dlambda-previous_iterate))
        iters=iters+1
    end
    return de_dlambda
end

"""
    dedbeta_complex(birth::Array,death::Array,migration::Array,i::Integer,eps::Number=1e-6,threshold::Number=1e-20)::Array

Return the derivative of e (the extinction probabilities) with respect to
β[i] (the birth rate of type i) using the complex step method.

Use the migration generator (Γ) calculated in the function
`migration_generator`.
In order to complete computations in single precision, change the 'eps'
parameter to 1f-6 (or similar).

# Example
```julia-repl
julia> dedbeta_complex([1.0;2.0],[2.0;1.0],[-1.0 1.0;1.0 -1.0],1)
2-element Array{Float64,1}:
 -0.08519781531172146
 -0.08997251113058993
```
"""
function dedbeta_complex(birth::Array,death::Array,migration::Array,i::Integer,eps::Number=1e-6,threshold::Number=1e-20)::Array
    birth=complex(birth)
    birth[i]=birth[i]+eps*im
    extinction=extinction_probability(birth,death,migration,threshold)
    return imag(extinction)./eps
end

"""
    deddelta_complex(birth::Array,death::Array,migration::Array,i::Integer,eps::Number=1e-6,threshold::Number=1e-20)::Array

Return the derivative of e (the extinction probabilities) with respect to
δ[i] (the death rate of type i) using the complex step method.

Use the migration generator (Γ) calculated in the function
`migration_generator`.
In order to complete computations in single precision, change the 'eps'
parameter to 1f-6 (or similar).

# Example
```julia-repl
julia> deddelta_complex([1.0;2.0],[2.0;1.0],[-1.0 1.0;1.0 -1.0],1)
2-element Array{Float64,1}:
 0.09595266340472737
 0.10133008745124994
```
"""
function deddelta_complex(birth::Array,death::Array,migration::Array,i::Integer,eps::Number=1e-6,threshold::Number=1e-20)::Array
    death=complex(death)
    death[i]=death[i]+eps*im
    extinction=extinction_probability(birth,death,migration,threshold)
    return imag(extinction)./eps
end

"""
    dedlambda_complex(birth::Array,death::Array,migration::Array,i::Integer,j::Integer,eps::Number=1e-6,threshold::Number=1e-20)::Array

Return the derivative of e (the extinction probabilities) with respect to
λ[i,j] (the migration rate from type i to type j where i is not equal
to j) using the complex step method.
In order to complete computations in single precision, change the 'eps'
parameter to 1f-6 (or similar).

# Example
```julia-repl
julia> dedlambda_complex([1.0;2.0],[2.0;1.0],[-1.0 1.0;1.0 -1.0],1,2)
2-element Array{Float64,1}:
 -0.10670751149777172
 -0.11268766377189804
```
"""
function dedlambda_complex(birth::Array,death::Array,migration::Array,i::Integer,j::Integer,eps::Number=1e-6,threshold::Number=1e-20)::Array
    migration=complex(migration)
    migration[i,j]=migration[i,j]+eps*im
    migration=migration_generator(migration)
    extinction=extinction_probability(birth,death,migration,threshold)
    return imag(extinction)./eps
end

"""
        dedtheta_inverse(birth::Array,death::Array,migration::Array, e::Array,dPdtheta::Array)::Array

Return the derivative of e (the extinction probabilities) with respect to a
parameter θ (β[i],δ[i],or λ[i,j]) using the inverse step method.  For the
dPdtheta parameter, use one of the dPddelta_inverse, dPdbeta_inverse,
dPdlambda_inverse functions.

# Example
```julia-repl
julia> dedtheta_inverse([1;2],[2;1],[-1 1;1 -1],extinction_probability([1;2],[2;1],[-1 1;1 -1]),dPddelta_inverse([1;2],[2;1],[-1 1;1 -1],1,extinction_probability([1;2],[2;1],[-1 1;1 -1])))
2-element Array{Float64,1}:
 0.0959526634047581
 0.10133008745127077
```
"""
function dedtheta_inverse(birth::Array,death::Array,migration::Array, e::Array,dPdtheta::Array)::Array
    n=size(migration)[1]
    dPde=zeros(n,n)
    for k=1:n
        for l=1:n
            if k!=l
                dPde[k,l]=migration[k,l]/(birth[k]+death[k]-migration[k,k])
            else
                dPde[k,l]=2*birth[k]*e[k]/(birth[k]+death[k]-migration[k,k])
            end
        end
    end
    return (inv(I-dPde)*dPdtheta)
end

"""
    drho_dtheta(generator::Array,domega_dtheta::Array)::Array

Return the derivative of the dominant eigenvalue of Ω (ρ) with respect to
θ (one of the rate parameters).

Use the generator (Ω) from `calculate_generator` and domega_dtheta from one of
the following the functions: `domega_dbeta` `domega_ddelta`, `domega_dlambda`.

# Example
```julia-repl
julia> drho_dtheta([-4.0 3.0;1.0 -2.0],[1.0 0.0;0.0 0.0])
1-element Array{Float64,1}:
 0.7500000000000002
```
"""
function drho_dtheta(omega::Array,domega_dtheta::Array)::Array
    n=size(omega,1)
    eigenvectors=normalized_eigenvectors(omega,n)
    return real((eigenvectors[1])*domega_dtheta*(eigenvectors[2]))
end

"""
    dmean_dtheta(omega::Array,domega_dtheta::Array, t::Number, tol::Number=1e-16)::Array

Return the derivative of the mean number of each type at time t with respect to
a parameter θ (β[i],δ[i], or λ[i,j]) using the functions for domega_dtheta
including domega_dbeta, domega_ddelta, and domega_dlambda.

# Example
```julia-repl
julia> dmean_dtheta(calculate_generator([1;2],[2;1],[-1 1;1 -1]),domega_ddelta(2,1),2)
2×2 Array{Float64,2}:
 -1.38934  0.203417
 -2.79707  0.793459
```
"""
function dmean_dtheta(omega::Array,domega_dtheta::Array, t::Number, tol::Number=1e-16)::Array
    return quadgk(s -> exp(s*omega)*domega_dtheta*exp((1-s)*omega), 0, t, rtol=tol)[1]
end

#=
Helper functions for Branching Process

This section includes functions called interally in the functions above or that
calculate important intermediate values for the branching process.
=#

"""
    migration_generator(migration::Array)::Array

Calculates the mean infinitesimal generator of the pure migration process (Γ).

Use to calculate the `migration` parameter used in the functions below.

# Examples
```julia-repl
julia> migration_generator([0 1 3;2 0 4;1 1 0])
3×3 Array{Int64,2}:
 -4   1   3
  2  -6   4
  1   1  -2
```
"""
function migration_generator(migration::Array)::Array
    return migration-Diagonal(vec(sum(migration,dims=2)))
end

"""
    calculate_generator(birth::Array,death::Array,migration::Array)::Array

Return the infinitesimal generator (Ω) of the birth-death-migration process with
the given rates.

Use to calculate the `generator` parameter for the functions below.
Use the migration infinitesimal generator (Γ) calculated in the function
`migration_generator`.

# Example
```julia-repl
julia> calculate_generator([1;2],[2;3],[-3 3;1 -1])
2×2 Array{Int64,2}:
 -4   3
  1  -2
```
"""
function calculate_generator(birth::Array,death::Array,migration::Array)::Array
    return Diagonal(birth)-Diagonal(death)+migration
end

"""
    calculate_offspring(birth::Array,death::Array,migration::Array)::Array

Return the offspring matrix (F) of the birth-death-migration process with
the given rates.

Use to calculate the `offspring` parameter for the functions below.
Use the migration generator (Γ) calculated in the function
`migration_generator`.

# Example
```julia-repl
julia> calculate_offspring([1;2],[2;3],[-3 3;1 -1])
2×2 Array{Float64,2}:
 0.333333  0.5
 0.166667  0.666667
```
"""
function calculate_offspring(birth::Array,death::Array,migration::Array)::Array
    return (migration-Diagonal(migration)+Diagonal(2*birth))./(birth+death-diag(migration))
end

"""
    dF_dbeta(birth::Array,death::Array,migration::Array,i::Integer)::Array

Return the derivative of F (the offspring matrix) with respect to β[i] (the
birth rate of type i).

Use the migration generator (Γ) calculated in the function
`migration_generator`.

# Example
```julia-repl
julia> dF_dbeta([1;2],[2;3],[-3 3;1 -1],1)
2×2 Array{Float64,2}:
 0.277778  -0.0833333
 0.0        0.0
  ```
"""
function dF_dbeta(birth::Array,death::Array,migration::Array,i::Integer)::Array
   n=length(birth)
   derivative=zeros(typeof(birth[1]),n,n)
   totalrate_squared=(birth[i]+death[i]-migration[i,i])^2
   derivative[i,:]=-migration[i,:]/totalrate_squared
   derivative[i,i]=2*(death[i]-migration[i,i])/totalrate_squared
   return derivative
end

"""
    dF_ddelta(birth::Array,death::Array,migration::Array,i::Integer)::Array

Return the derivative of F (the offspring matrix) with respect to δ[i] (the
death rate of type i).

Use the migration generator (Γ) calculated in the function
`migration_generator`.

# Example
```julia-repl
julia> dF_ddelta([1;2],[2;3],[-3 3;1 -1],1)
2×2 Array{Float64,2}:
 -0.0555556  -0.0833333
  0.0         0.0
  ```
"""
function dF_ddelta(birth::Array,death::Array,migration::Array,i::Integer)::Array
   n=length(birth)
   derivative=zeros(typeof(birth[1]),n,n)
   totalrate_squared=(birth[i]+death[i]-migration[i,i])^2
   derivative[i,:]=-migration[i,:]/totalrate_squared
   derivative[i,i]=-2*birth[i]/totalrate_squared
   return derivative
end

"""
   dF_dlambda(birth::Array,death::Array,migration::Array,i::Integer,j::Integer)::Array

Return the derivative of F (the offspring matrix) with respect to λ[i,j]
(the migration rate from type i to type j where i is not equal to j).

Use the migration generator (Γ) calculated in the function
`migration_generator`.

# Example
```julia-repl
julia> dF_dlambda([1;2;1],[2;3;2],[-3 2 1;1 -2 1;1 3 -4],1,2)
3×3 Array{Float64,2}:
-0.0555556  0.111111  -0.0277778
 0.0        0.0        0.0
 0.0        0.0        0.0
 ```
"""
function dF_dlambda(birth::Array,death::Array,migration::Array,i::Integer,j::Integer)::Array
   n=length(birth)
   derivative=zeros(typeof(birth[1]),n,n)
   totalrate=birth[i]+death[i]-migration[i,i]
   totalrate_squared=totalrate^2
   derivative[i,:]=-migration[i,:]/totalrate_squared
   derivative[i,i]=-2*birth[i]/totalrate_squared
   derivative[i,j]=(totalrate-migration[i,j])/totalrate_squared
   return derivative
end

"""
    domega_dbeta(n::Integer,i::Integer,singlePrecision::Bool=false)::Array

Return the derivative of Ω nxn with respect to β[i] (the rate of birth of
type i).

# Example
```julia-repl
julia> domega_dbeta(2,1)
2×2 Array{Float64,2}:
 1.0  0.0
 0.0  0.0
```
"""
function domega_dbeta(n::Integer,i::Integer,singlePrecision::Bool=false)::Array
    if singlePrecision
        domega_dbeta=zeros(Float32,n,n)
    else
        domega_dbeta=zeros(n,n)
    end
    domega_dbeta[i,i]=1
    return domega_dbeta
end

"""
    domega_ddelta(n::Integer,i::Integer,singlePrecision::Bool=false)::Array

Return the derivative of Ω nxn with respect to δ[i] (the rate of death of
type i).

# Example
```julia-repl
julia> domega_ddelta(2,1)
2×2 Array{Float64,2}:
 -1.0  0.0
  0.0  0.0
```
"""
function domega_ddelta(n::Integer,i::Integer,singlePrecision::Bool=false)::Array
    if singlePrecision
        domega_ddelta=zeros(Float32,n,n)
    else
        domega_ddelta=zeros(n,n)
    end
    domega_ddelta[i,i]=-1
    return domega_ddelta
end

"""
    domega_dlambda(n::Integer,i::Integer,singlePrecision::Bool=false)::Array

Return the derivative of Ω nxn with respect to λ[i,j] (the rate of
migration of type i to type j).

# Example
```julia-repl
julia> domega_dlambda(3,1,2)
3×3 Array{Float64,2}:
 -1.0  1.0  0.0
  0.0  0.0  0.0
  0.0  0.0  0.0
```
"""
function domega_dlambda(n::Integer,i::Integer,j::Integer,singlePrecision::Bool=false)::Array
    if singlePrecision
        domega_dlambda=zeros(Float32,n,n)
    else
        domega_dlambda=zeros(n,n)
    end
    domega_dlambda[i,i]=-1
    domega_dlambda[i,j]=1
    return domega_dlambda
end

"""
    find_index(generator::Array,n::Integer)::Integer

Find the index of the dominant eigenvalue of the generator matrix.

# Example
```julia-repl
julia> find_index([-4.0 3.0;1.0 -2.0],2)
1
```
"""
function find_index(generator::Array,n::Integer)::Integer
    index=1
    E=eigvals(generator)
    maxeigen=E[1]
    for i=2:n
        val=E[i]
        if abs(real(val))>abs(real(maxeigen))
            maxeigen=val
            index=i
        end
    end
    return index
end

"""
    find_rho(generator::Array,n::Integer)::Integer

Find the dominant eigenvalue of the generator matrix.

# Example
```julia-repl
julia> find_rho([-4.0 3.0;1.0 -2.0],2)
-5.0
```
"""
function find_rho(generator::Array,n::Integer)::Number
    E=eigvals(generator)
    maxeigen=E[1]
    for i=2:n
        val=E[i]
        if abs(real(val))>abs(real(maxeigen))
            maxeigen=val
        end
    end
    return maxeigen
end

"""
    function normalized_eigenvectors(generator::Array,n::Integer)::Tuple

Find the (left,right) eigenvectors of the nth eigenvalue of the generator
normalized such that their product = 1.

# Example
```julia-repl
julia> normalized_eigenvectors([-4.0 3.0;1.0 -2.0],2)
([-0.7476743906106104 0.7476743906106104], [-1.0031104574646332, 0.33437015248821106])
```
"""
function normalized_eigenvectors(generator::Array,n::Integer)::Tuple
    index=find_index(generator,n)
    left_eigen=dominant_left(generator,index)
    right_eigen=dominant_right(generator,index)
    normalizer=normalization_factor(left_eigen,right_eigen)
    return (normalizer*left_eigen,normalizer*right_eigen)
end

"""
    function dominant_right(generator::Array,index::Integer)::Array

Find the right eigenvector of the eigenvalue of the generator at the index
specified.

# Example
```julia-repl
julia> dominant_right([-4.0 3.0;1.0 -2.0],1)
2-element Array{Float64,1}:
 -0.9486832980505138
  0.31622776601683794
```
"""
function dominant_right(generator::Array,index::Integer)::Array
    return eigvecs(generator)[:,index]
end

"""
    function dominant_left(generator::Array,index::Integer)::Array

Find the left eigenvector of the eigenvalue of the generator at the index
specified.

# Example
```julia-repl
julia> dominant_left([-4.0 3.0;1.0 -2.0],1)
1×2 Array{Float64,2}:
 -0.707107  0.707107
```
"""
function dominant_left(generator::Array,index::Integer)::Array
    return transpose(eigvecs(transpose(generator))[:,index])
end

"""
    function normalization_factor(left_eigen::Array,right_eigen::Array)::Number

Find the constant by which it is required to multiply of each vector such that
left*right=1.

# Example
```julia-repl
julia> normalization_factor([-.707 .707],[-.949,.316])
1.0574139372940603
```
"""
function normalization_factor(left_eigen::Array,right_eigen::Array)::Number
    return 1/sqrt(Complex((left_eigen*right_eigen)[1]))
end

"""
    function dPdbeta_inverse(birth::Array,death::Array,migration::Array,i::Integer, e::Array)::Array

Find the derivative of P (the vector extinction equation) with respect to β[i].

# Example
```julia-repl
julia> dPdbeta_inverse([1;2],[2;1],[-1 1;1 -1],1,extinction_probability([1;2],[2;1],[-1 1;1 -1]))
2-element Array{Float64,1}:
 -0.024880475692082057
  0.0
```
"""
function dPdbeta_inverse(birth::Array,death::Array,migration::Array,i::Integer, e::Array)::Array
    n=size(migration)[1]
    dPdbeta=zeros(n)
    totalrate_squared=(birth[i]+death[i]-migration[i,i])^2
    dPdbeta[i]=-death[i]/totalrate_squared+e[i]^2*(death[i]-migration[i,i])/totalrate_squared
    for j=1:n
        if j!=i
            dPdbeta[i]=dPdbeta[i]-migration[i,j]*e[j]/totalrate_squared
        end
    end
    return dPdbeta
end

"""
    function dPddelta_inverse(birth::Array,death::Array,migration::Array,i::Integer, e::Array)::Array

Find the derivative of P (the vector extinction equation) with respect to δ[i].

# Example
```julia-repl
julia> dPddelta_inverse([1;2],[2;1],[-1 1;1 -1],1,extinction_probability([1;2],[2;1],[-1 1;1 -1]))
2-element Array{Float64,1}:
 0.028021233886074345
 0.0
```
"""
function dPddelta_inverse(birth::Array,death::Array,migration::Array,i::Integer, e::Array)::Array
    n=size(migration)[1]
    dPddelta=zeros(n)
    totalrate_squared=(birth[i]+death[i]-migration[i,i])^2
    dPddelta[i]=(birth[i]-migration[i,i])/totalrate_squared-birth[i]*e[i]^2/totalrate_squared
    for j=1:n
        if j!=i
            dPddelta[i]=dPddelta[i]-migration[i,j]*e[j]/totalrate_squared
        end
    end
    return dPddelta
end

"""
    function dPdlambda_inverse(birth::Array,death::Array,migration::Array,i::Integer, j::Integer, e::Array)::Array

Find the derivative of P (the vector extinction equation) with respect to
λ[i,j].

# Example
```julia-repl
julia> dPdlambda_inverse([1;2],[2;1],[-1 1;1 -1],1,2,extinction_probability([1;2],[2;1],[-1 1;1 -1]))
2-element Array{Float64,1}:
 -0.03116199208006662
  0.0
```
"""
function dPdlambda_inverse(birth::Array,death::Array,migration::Array,i::Integer, j::Integer, e::Array)::Array
    n=size(migration)[1]
    dPdlambda=zeros(n)
    totalrate_squared=(birth[i]+death[i]-migration[i,i])^2
    dPdlambda[i]=-death[i]/totalrate_squared-birth[i]*e[i]^2/totalrate_squared+(birth[i]+death[i]-migration[i,i]-migration[i,j])*e[j]/totalrate_squared
    for k=1:n
        if (k!=i)&(k!=j)
            dPdlambda[i]=dPdlambda[i]-migration[i,k]*e[k]/totalrate_squared
        end
    end
    return dPdlambda
end

"""
    function expected_number(generator::Array,t::Number)::Array

Find the expected number of each type at time t.

# Example
```julia-repl
julia> expected_number(calculate_generator([1;2],[2;1],[-1 1;1 -1]),2)
2×2 Array{Float64,2}:
 0.342149  0.806708
 0.806708  1.95556
```
"""
function expected_number(generator::Array,t::Number)::Array
    return exp(generator*t)
end

#=
SIR Model

This section includes functions to compute the following quantities for
SIR Model using the following methods:
    M (mean total number of infecteds)
        iterative method
    T (mean time to extinction)
        iterative method
    dM/dθ (where θ in {δ,η})
        iterative method
        complex step method
    dT/dθ (where θ in {δ,η})
        iterative method
        complex step method
    S,I,R
        direct
    (dS,dI,dR)/dθ (where θ in {δ,η})
        direct
        complex
=#

"""
    function meannumber(N::Integer,delta::Number,eta::Number)::Array

Return the mean number ever infected M where N is the total number in the
population, and M[i+1] is the mean number ever infected starting from i
infected individuals.

# Example
```julia-repl
julia> meannumber(3,1,1)
4-element Array{Float64,1}:
 0.0
 1.5750000000000002
 2.4375
 3.0
```
"""
function meannumber(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::Array
    return meannumber_matrix(N,delta,eta,TP)[:,N+1]
end

"""
    function meantime(N::Integer,delta::Number,eta::NumberTP::Type=Float64)::UpperTriangular

Return the mean time to elimination T where N is the total number in the
population, and T[i+1] is the mean time to elimination starting from i
infected individuals.

# Example
```julia-repl
julia> meantime(3,1,1)
4-element Array{Float64,1}:
 0.0
 1.2708333333333335
 1.6770833333333335
 1.8333333333333333
```
"""
function meantime(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::Array
    return meantime_matrix(N,delta,eta,TP)[:,N+1]
end

"""
    function dm_ddelta(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::Array

Using the direct differentiation method, return the derivative of the mean
number ever infected M with respect to δ where N is the total number in the
population, and dM[i+1] is the derivative for the process starting from i
infected individuals.

# Example
```julia-repl
julia> dm_ddelta(3,1,1)
4-element Array{Float64,1}:
  0.0
 -0.4575000000000001
 -0.28125
  0.0
```
"""
function dm_ddelta(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::Array
    M=meannumber_matrix(N,delta,eta,TP)
    dM=UpperTriangular(zeros(TP,N+1,N+1))
    for n in 1:N
        for j in 0:n-1
            i=n-j
            if i!=n
                immunityrate=i*delta
                infectionproportion=i*(n-i)/N
                totalrate=1/(immunityrate+infectionproportion*eta)
                ratesquared=totalrate^2
                dM[i+1,n+1]=(M[i,n]+1-M[i+2,n+1])*i*infectionproportion*eta*ratesquared+immunityrate*totalrate*dM[i,n]+infectionproportion*eta*totalrate*dM[i+2,n+1]
            end
        end
    end
    return dM[:,N+1]
end

"""
    function dm_deta(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::Array

Using the direct differentiation method, return the derivative of the mean
number ever infected M with respect to η where N is the total number in the
population, and dM[i+1] is the derivative for the process starting from i
infected individuals.

# Example
```julia-repl
julia> dm_deta(3,1,1)
4-element Array{Float64,1}:
 0.0
 0.4575000000000001
 0.28125
 0.0
```
"""
function dm_deta(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::Array
    M=meannumber_matrix(N,delta,eta,TP)
    dM=UpperTriangular(zeros(TP,N+1,N+1))
    for n in 1:N
        for j in 0:n-1
            i=n-j
            if i!=n
                immunityrate=i*delta
                infectionproportion=i*(n-i)/N
                totalrate=1/(immunityrate+infectionproportion*eta)
                ratesquared=totalrate^2
                dM[i+1,n+1]=(M[i+2,n+1]-M[i,n]-1)*immunityrate*infectionproportion*ratesquared+immunityrate*totalrate*dM[i,n]+infectionproportion*eta*totalrate*dM[i+2,n+1]
            end
        end
    end
    return dM[:,N+1]
end

"""
    function dt_ddelta(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::Array

Using the direct differentiation method, return the derivative of the mean time
to elimination T with respect to δ where N is the total number in the
population, and dT[i+1] is the derivative for the process starting from i
infected individuals.

# Example
```julia-repl
julia> dt_ddelta(3,1,1)
4-element Array{Float64,1}:
  0.0
 -1.4770833333333335
 -1.786458333333333
 -1.8333333333333333
```
"""
function dt_ddelta(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::Array
    T=meantime_matrix(N,delta,eta,TP)
    dT=UpperTriangular(zeros(TP,N+1,N+1))
    for n in 1:N
        for j in 0:n-1
            i=n-j
            immunityrate=i*delta
            if i==n
                dT[i+1,n+1]=dT[i,n]-1/(delta*immunityrate)
            else
                infectionrate=i*(n-i)*eta/N
                totalrate=1/(immunityrate+infectionrate)
                dT[i+1,n+1]=totalrate*(-i*totalrate+i*infectionrate*totalrate*(T[i,n]-T[i+2,n+1])+immunityrate*dT[i,n]+infectionrate*dT[i+2,n+1])
            end
        end
    end
    return dT[:,N+1]
end

"""
    function dt_deta(N::Integer,delta::Number,eta::Number,TP::Type=Float64)

Using the direct differentiation method, return the derivative of the mean time
to elimination T with respect to η where N is the total number in the
population, and dT[i+1] is the derivative for the process starting from i
infected individuals.

# Example
```julia-repl
julia> dt_deta(3,1,1)
4-element Array{Float64,1}:
 0.0
 0.2062500000000001
 0.109375
 0.0
```
"""
function dt_deta(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::Array
    T=meantime_matrix(N,delta,eta,TP)
    dT=UpperTriangular(zeros(TP,N+1,N+1))
    for n in 1:N
        for j in 0:n-1
            i=n-j
            if i!=n
                immunityrate=i*delta
                infectionproportion=i*(n-i)/N
                totalrate=1/(immunityrate+infectionproportion*eta)
                ratesquared=totalrate^2
                dT[i+1,n+1]=-infectionproportion*ratesquared+immunityrate*totalrate*dT[i,n]+infectionproportion*eta*totalrate*dT[i+2,n+1]-immunityrate*infectionproportion*ratesquared*T[i,n]+infectionproportion*immunityrate*ratesquared*T[i+2,n+1]
            end
        end
    end
    return dT[:,N+1]
end

"""
    function dmddelta_complex(N::Integer,delta::Number,eta::Number,TP::Type=Float64,eps::Number=1e-6)::Array

Using the complex step method, return the derivative of the mean number ever
infected M with respect to δ where N is the total number in the population,
and dM[i+1] is the derivative for the process starting from i infected
individuals.

# Example
```julia-repl
julia> dmddelta_complex(3,1,1)
4-element Array{Float64,1}:
  0.0
 -0.45749999999975116
 -0.2812499999998948
  0.0
```
"""
function dmddelta_complex(N::Integer,delta::Number,eta::Number,TP::Type=Float64,eps::Number=1e-6,)::Array
    CTP=Complex{TP}
    delta=complex(delta)+eps*im
    M=meannumber(N,delta,eta,CTP)
    return imag(M)./eps
end

"""
    function dmdeta_complex(N::Integer,delta::Number,eta::Number,eps::Number=1e-6, TP::Type=Float64)::Array

Using the complex step method, return the derivative of the mean number ever
infected M with respect to η where N is the total number in the population,
and dM[i+1] is the derivative for the process starting from i infected
individuals.

# Example
```julia-repl
julia> dmdeta_complex(3,1,1)
4-element Array{Float64,1}:
 0.0
 0.45749999999998303
 0.2812499999999648
 0.0
```
"""
function dmdeta_complex(N::Integer,delta::Number,eta::Number,eps::Number=1e-6, TP::Type=Float64)::Array
    CTP=Complex{TP}
    eta=complex(eta)+eps*im
    M=meannumber(N,delta,eta,CTP)
    return imag(M)./eps
end

"""
    function dtddelta_complex(N::Integer,delta::Number,eta::Number,TP::Type=Float64,eps::Number=1e-6)::Array

Using the complex step method, return the derivative of the mean time to
elimination T with respect to δ where N is the total number in the
population, and dT[i+1] is the derivative for the process starting from i
infected individuals.

# Example
```julia-repl
julia> dtddelta_complex(3,1,1)
4-element Array{Float64,1}:
  0.0
 -1.477083333331603
 -1.7864583333314472
 -1.8333333333314998
```
"""
function dtddelta_complex(N::Integer,delta::Number,eta::Number,TP::Type=Float64,eps::Number=1e-6)::Array
    CTP=Complex{TP}
    delta=complex(delta)+eps*im
    T=meantime(N,delta,eta,CTP)
    return imag(T)./eps
end

"""
    function dtdeta_complex(N::Integer,delta::Number,eta::Number,TP::Type=Float64,eps::Number=1e-6)::Array

Using the complex step method, return the derivative of the mean time to
elimination T with respect to η where N is the total number in the population,
and dT[i+1] is the derivative for the process starting from i infected
individuals.

# Example
```julia-repl
julia> dt_deta(3,1,1)
4-element Array{Float64,1}:
 0.0
 0.2062500000000001
 0.109375
 0.0
```
"""
function dtdeta_complex(N::Integer,delta::Number,eta::Number,eps::Number=1e-6, TP::Type=Float64)::Array
    CTP=Complex{TP}
    eta=complex(eta)+eps*im
    T=meantime(N,delta,eta,CTP)
    return imag(T)./eps
end

"""
    solveSystem(system::Function,u0::Array,p::Array,tspan::Tuple)::Array

Solves a system of ODEs and returns the values of each quantity at the last time
point.

# Example
```julia-repl
julia> solveSystem(SIR!,[9;1;0],[.5;.5;10],(0.0,100.0))
3-element Array{Float64,1}:
 6.083432574396499
 9.817618881395788e-9
 3.916567415785882
```
"""
function solveSystem(system::Function,u0::Array,p::Array,tspan::Tuple)::Array
    prob = ODEProblem(system,u0,tspan,p)
    sol = solve(prob)
    return sol(tspan[2])
end

"""
    solveSystem_complex(system::Function,u0::Array,p::Array,tspan::Tuple,derivParam::Integer,eps::Number=1e-6)::Array

Uses the complex step method to solve for the derivatives of a system of ODEs
with respect to a parameter (the parameter at index derivParam of p).

# Example
```julia-repl
julia> solveSystem_complex(SIR!,[9.0;1.0;0.0],[.5;.5;10.0],(0.0,100.0),1)
3-element Array{Float64,1}:
 -12.166825657544354
   4.2329669591818346e-8
  12.166825615214691
```
"""
function solveSystem_complex(system::Function,u0::Array,p::Array,tspan::Tuple,derivParam::Integer,eps::Number=1e-6)::Array
    p=complex(p)
    u0=complex(u0)
    p[derivParam]=p[derivParam]+eps*im
    prob = ODEProblem(system,u0,tspan,p)
    sol = solve(prob)
    return imag(sol(tspan[2]))./eps
end

#=
This section contains helper functions for the SIR model methods above.
=#

"""
    function meannumber_matrix(N::Integer,delta::Number,eta::Number)::UpperTriangular

This function returns the entire recursively derived matrix created to
calculate  M.

# Example
```julia-repl
julia> meannumber_matrix(3,1,1)
4×4 UpperTriangular{Float64,Array{Float64,2}}:
 0.0  0.0  0.0   0.0
  ⋅   1.0  1.25  1.575
  ⋅    ⋅   2.0   2.4375
  ⋅    ⋅    ⋅    3.0
```
"""
function meannumber_matrix(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::UpperTriangular
    M=UpperTriangular(zeros(TP,N+1,N+1))
    for n in 1:N
        for j in 0:n-1
            i=n-j
            if i==n
                M[i+1,n+1]=i
            else
                immunityrate=i*delta
                infectionrate=i*(n-i)*eta/N
                totalrate=1/(immunityrate+infectionrate)
                M[i+1,n+1]=immunityrate*totalrate*(M[i,n]+1)+infectionrate*totalrate*M[i+2,n+1]
            end
        end
    end
    return M
end

"""
    function meantime_matrix(N::Integer,delta::Number,eta::Number)::UpperTriangular

This function returns the entire recursively derived matrix created to
calculate  T.

# Example
```julia-repl
julia> meantime_matrix(3,1,1)
4×4 UpperTriangular{Float64,Array{Float64,2}}:
 0.0  0.0  0.0    0.0
  ⋅   1.0  1.125  1.27083
  ⋅    ⋅   1.5    1.67708
  ⋅    ⋅    ⋅     1.83333
```
"""
function meantime_matrix(N::Integer,delta::Number,eta::Number,TP::Type=Float64)::UpperTriangular
    T=UpperTriangular(zeros(TP,N+1,N+1))
    for n in 1:N
        for j in 0:n-1
            i=n-j
            immunityrate=i*delta
            if i==n
                T[i+1,n+1]=T[i,i]+1/immunityrate
            else
                infectionrate=i*(n-i)*eta/N
                totalrate=1/(immunityrate+infectionrate)
                T[i+1,n+1]=totalrate*(1+immunityrate*T[i,n]+infectionrate*T[i+2,n+1])
            end
        end
    end
    return T
end

"""
    SIR!(du,u,p,t)

This function contains the system of equations for the SIR model.
"""
function SIR!(du,u,p,t)
    S,I,R = u
    η,δ,N = p
    du[1] = -η*S*I/N #dS/dt
    du[2] = η*S*I/N - δ*I #dI/dt
    du[3] = δ*I #dR/dt
end

"""
    SIRDerivs!(du,u,p,t)

This function contains the system of equations for the SIR model derivatives.
"""
function SIRDerivs!(du,u,p,t)
    S,I,R,dSdη,dIdη,dRdη,dSdδ,dIdδ,dRdδ = u
    η,δ,N = p
    du[1] = -η*S*I/N #dS/dt
    du[2] = η*S*I/N - δ*I #dI/dt
    du[3] = δ*I #dR/dt
    du[4] = -I*S/N-η*dIdη*S/N-η*I*dSdη/N #d^2S/dtdη
    du[5] = I*S/N+η*dIdη*S/N+η*I*dSdη/N-δ*dIdη #d^2I/dtdη
    du[6] = δ*dIdη #d^2R/dtdη
    du[7] = -η*dIdδ*S/N-η*I*dSdδ/N #d^2S/dtdδ
    du[8] = η*dIdδ*S/N+η*I*dSdδ/N-I-δ*dIdδ #d^2S/dtdδ
    du[9] = I+δ*dIdδ #d^2S/dtdδ
end
