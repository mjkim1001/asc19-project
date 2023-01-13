import Base: iterate
using Printf, IterativeSolvers, LinearAlgebra, Random, Distributions
export cg, cg!, CGSampler, PCGSampler, cgSamp!, CGStateVariables

"""
  The output of cgSamp!() is
  y ~ N(A^(-1)b , A^(-1))
  x = A^(-1)b
"""
mutable struct CGSampler{matT, solT, vecT, numT <: Real}
    A::matT
    x::solT
    r::vecT
    c::vecT
    u::vecT
    y::vecT
    α::Float64
    a::Array{Float64,1}
    b::Array{Complex,1}
    reltol::numT
    residual::numT
    prev_residual::numT
    maxiter::Int
    mv_products::Int
end

mutable struct PCGSampler{precT, matT, solT, vecT, numT <: Real, paramT <: Number}
    Pl::precT
    A::matT
    x::solT
    r::vecT
    c::vecT
    u::vecT
    y::vecT
    α::Float64
    a::Array{Float64,1}
    b::Array{Complex,1}
    reltol::numT
    residual::numT
    ρ::paramT
    maxiter::Int
    mv_products::Int
end

@inline converged(it::Union{CGSampler, PCGSampler}) = it.residual ≤ it.reltol

@inline start(it::Union{CGSampler, PCGSampler}) = 0

@inline done(it::Union{CGSampler, PCGSampler}, iteration::Int) = iteration ≥ it.maxiter || converged(it)


###############
# Ordinary CG #
###############

function iterate(it::CGSampler, iteration::Int=start(it))
    if done(it, iteration) return nothing end
    # u := r + βu (almost an axpy)
    β = it.residual^2 / it.prev_residual^2
    if(iteration != 0) 
        append!( it.a, - β/it.α)
        append!( it.b, sqrt(complex(-β))/it.α )
    end
    it.u .= it.r .+ β .* it.u

    # c = A * u
    mul!(it.c, it.A, it.u)
    α = it.residual^2 / dot(it.u, it.c)
    it.a[iteration+1] =  it.a[iteration+1] + 1/it.α
    # Improve solution and residual
    it.x .+= α .* it.u
    it.y = it.y + randn() .* it.u ./sqrt(dot(it.u, it.c)) 
    it.r .-= α .* it.c
    
    it.prev_residual = it.residual
    it.residual = norm(it.r)

    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end

#####################
# Preconditioned CG #
#####################

function iterate(it::PCGSampler, iteration::Int=start(it))
    # Check for termination first
    print(iteration)
    if done(it, iteration)
        return nothing
    end
    ## c = Pl \ r ##
    ldiv!(it.c, it.Pl, it.r)

    ρ_prev = it.ρ
    it.ρ = dot(it.c, it.r)

    # u := c + βu (almost an axpy)
    β = it.ρ / ρ_prev
    if(iteration != 0) 
        append!( it.a, - β/it.α)
        append!( it.b, sqrt(complex(-β))/it.α )
    end
    it.u .= it.c .+ β .* it.u

    # c = A * u
    mul!(it.c, it.A, it.u)
    it.α = it.ρ / dot(it.u, it.c)
    it.a[iteration+1] =  it.a[iteration+1] + 1/it.α
    
    # Improve solution and residual
    it.x .+= it.α .* it.u
    it.y = it.y + randn() .* it.u ./sqrt(dot(it.u, it.c)) 
    it.r .-= it.α .* it.c

    it.residual = norm(it.r)

    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end


struct CGStateVariables{T,Tx<:AbstractArray{T}}
    u::Tx
    r::Tx
    c::Tx
end

function cg_sampler!(x, A, y0, b, Pl = Identity();
    tol = sqrt(eps(real(eltype(b)))),
    maxiter::Int = size(A, 2),
    statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
    initially_zero::Bool = false
)
    u = statevars.u
    r = statevars.r
    c = statevars.c
    u .= zero(eltype(x))
    copyto!(r, b)

    # Compute r with an MV-product or not.
    if initially_zero
        mv_products = 0
        c = similar(x)
        residual = norm(b)
        reltol = residual * tol # Save one dot product
    else
        mv_products = 1
        mul!(c, A, x)
        r .-= c
        residual = norm(r)
        reltol = norm(b) * tol
    end

    # Return the Sampler
    if isa(Pl, Identity)
        return CGSampler(A, x, r, c, u, y0, 0., zeros(1), zeros(Complex,0),
            reltol, residual, one(residual),
            maxiter, mv_products
        )
    else
        return PCGSampler(Pl, A, x, r, c, u, y0, 0., zeros(1), zeros(Complex,0),
            reltol, residual, one(eltype(x)),
            maxiter, mv_products
        )
    end
end



function cgSamp!(x, A, y0, b;
    tol = sqrt(eps(real(eltype(b)))),
    maxiter::Int = size(A, 2),
    log::Bool = false,
    statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
    verbose::Bool = false,
    Pl = Identity(),
    kwargs...
)
    history = ConvergenceHistory(partial = !log)
    history[:tol] = tol
    log && reserve!(history, :resnorm, maxiter + 1)

    # Actually perform CG
    iterable = cg_sampler!(x, A, y0, b, Pl; tol = tol, maxiter = maxiter, statevars = statevars, kwargs...)
    if log
        history.mvps = iterable.mv_products
    end
    for (iteration, item) = enumerate(iterable)
        if log
            nextiter!(history, mvps = 1)
            push!(history, :resnorm, iterable.residual)
        end
        verbose && @printf("%3d\t%1.2e\n", iteration, iterable.residual)
    end

    verbose && println()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (iterable.x, history) : iterable
end

"""
Use like this

b = randn(n^2)
x = zeros(n^2)
L = LowerTriangular(A)
p = CholeskyPreconditioner(L, 2)
samp = cgSamp!(x, A, x, b;
    tol = sqrt(eps(real(eltype(b)))),
    maxiter= 50,
    Pl = p)

"""