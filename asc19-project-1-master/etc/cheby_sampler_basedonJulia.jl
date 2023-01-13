#########################
# This code is our Try-1 for Cheby-accelerated iteration solver and sampler. We referred to the julia code for sor and ssor and based on that, implemented our one chebyshev accelerated iteration solver and sampler, which performed very well.
#########################

import LinearAlgebra: mul!, ldiv!
import Base: getindex, iterate
using SparseArrays, Arpack, LinearAlgebra
using Distributions, BenchmarkTools, IterativeSolvers, MatrixDepot, Random

"""
Basic structure for Sparse matrices
"""
struct DiagonalIndices{Tv, Ti <: Integer}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::Vector{Ti}

    function DiagonalIndices{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
        # Check square?
        diag = Vector{Ti}(undef, A.n)

        for col = 1 : A.n
            r1 = Int(A.colptr[col])
            r2 = Int(A.colptr[col + 1] - 1)
            r1 = searchsortedfirst(A.rowval, col, r1, r2, Base.Order.Forward)
            if r1 > r2 || A.rowval[r1] != col || iszero(A.nzval[r1])
                throw(LinearAlgebra.SingularException(col))
            end
            diag[col] = r1
        end 

        new(A, diag) #
    end
end

DiagonalIndices(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = DiagonalIndices{Tv,Ti}(A)
@inline getindex(d::DiagonalIndices, i::Int) = d.diag[i]


struct FastLowerTriangular{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end

struct FastUpperTriangular{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end

struct StrictlyUpperTriangular{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end

struct StrictlyLowerTriangular{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end

struct OffDiagonal{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end


"""
Forward substitution for the FastLowerTriangular type
"""
function forward_sub!(F::FastLowerTriangular, x::AbstractVector)
    A = F.matrix

    @inbounds for col = 1 : A.n

        # Solve for diagonal element
        idx = F.diag[col]
        x[col] /= A.nzval[idx]

        # Substitute next values involving x[col]
        for i = idx + 1 : (A.colptr[col + 1] - 1)
            x[A.rowval[i]] -= A.nzval[i] * x[col]
        end
    end

    x
end

"""
Forward substitution
"""
function forward_sub!(α, F::FastLowerTriangular, x::AbstractVector, β, y::AbstractVector)
    A = F.matrix

    @inbounds for col = 1 : A.n

        # Solve for diagonal element
        idx = F.diag[col]
        x[col] = α * x[col] / A.nzval[idx] + β * y[col]

        # Substitute next values involving x[col]
        for i = idx + 1 : (A.colptr[col + 1] - 1)
            x[A.rowval[i]] -= A.nzval[i] * x[col]
        end
    end

    x
end

"""
Backward substitution for the FastUpperTriangular type
"""
function backward_sub!(F::FastUpperTriangular, x::AbstractVector)
    A = F.matrix

    @inbounds for col = A.n : -1 : 1

        # Solve for diagonal element
        idx = F.diag[col]
        x[col] = x[col] / A.nzval[idx]

        # Substitute next values involving x[col]
        for i = A.colptr[col] : idx - 1
            x[A.rowval[i]] -= A.nzval[i] * x[col]
        end
    end

    x
end

function backward_sub!(α, F::FastUpperTriangular, x::AbstractVector, β, y::AbstractVector)
    A = F.matrix

    @inbounds for col = A.n : -1 : 1

        # Solve for diagonal element
        idx = F.diag[col]
        x[col] = α * x[col] / A.nzval[idx] + β * y[col]

        # Substitute next values involving x[col]
        for i = A.colptr[col] : idx - 1
            x[A.rowval[i]] -= A.nzval[i] * x[col]
        end
    end

    x
end


"""
------------------------------------------------------------------------------
SSOR start
------------------------------------------------------------------------------
"""

mutable struct SSORIterable{T, solT, vecT, rhsT, numT <: Real}
    sL::StrictlyLowerTriangular
    sU::StrictlyUpperTriangular
    L::FastLowerTriangular
    U::FastUpperTriangular
    ω::numT
    x::solT
    tmp::vecT
    b::rhsT
    maxiter::Int
end

function ssor_iterable(x::AbstractVector{T}, A::SparseMatrixCSC, b::AbstractVector, ω::Real; maxiter::Int = 10) where {T}
    D = DiagonalIndices(A)
    SSORIterable{T,typeof(x),typeof(x),typeof(b),typeof(ω)}(
        StrictlyLowerTriangular(A, D),
        StrictlyUpperTriangular(A, D),
        FastLowerTriangular(A, D),
        FastUpperTriangular(A, D),
        ω, x, similar(x), b, maxiter
    )
end

start(s::SSORIterable) = 1
done(s::SSORIterable, iteration::Int) = iteration > s.maxiter

function iterate(s::SSORIterable{T}, iteration::Int=start(s)) where {T}
    if done(s, iteration) return nothing end

    # tmp = b - U * x
    gauss_seidel_multiply!(-one(T), s.sU, s.x, one(T), s.b, s.tmp)

    # tmp = ω * inv(L) * tmp + (1 - ω) * x
    forward_sub!(s.ω, s.L, s.tmp, one(T) - s.ω, s.x)

    # x = b - L * tmp
    gauss_seidel_multiply!(-one(T), s.sL, s.tmp, one(T), s.b, s.x)

    # x = ω * inv(U) * x + (1 - ω) * tmp
    backward_sub!(s.ω, s.U, s.x, one(T) - s.ω, s.tmp)

    nothing, iteration + 1
end


"""
ssor!(x, A::SparseMatrixCSC, b, ω::Real; maxiter=10)
Performs exactly `maxiter` SSOR iterations with relaxation parameter `ω`. Each iteration
is basically a forward *and* backward sweep of SOR.
Allocates a temporary vector and precomputes the diagonal indices.
Throws `LinearAlgebra.SingularException` when the diagonal has a zero. This check
is performed once beforehand.
"""

function ssor!(x::AbstractVector, A::SparseMatrixCSC, b::AbstractVector, ω::Real; maxiter::Int = 10)
    iterable = ssor_iterable(x, A, b, ω, maxiter = maxiter)
    for item = iterable end
    iterable.x
end

"""
------------------------------------------------------------------------------
Cheby-ssor Iteration Solver start
------------------------------------------------------------------------------
"""

mutable struct CSSORIterable{T, solT, vecT, rhsT, numT <: Real}
    sL::StrictlyLowerTriangular
    sU::StrictlyUpperTriangular
    L::FastLowerTriangular
    U::FastUpperTriangular
#    Dw::Diagonal
    δ::numT
    τ::numT
    β::numT
    α::numT
    b::numT
    a::numT
    κ::numT
    ω::numT
    x::solT
    tmp::vecT
    prev::vecT
    w::vecT
    d::rhsT
    maxiter::Int
end


function cheby_ssor_iterable(x::AbstractVector{T}, A::SparseMatrixCSC, d::AbstractVector, ω::Real, λ_max::Real, λ_min::Real; maxiter::Int = 10) where {T}
    D = DiagonalIndices(A)
    #Dw = sqrt((2/ω-1)) * Diagonal(sqrt.(diag(A)))
    
    
    δ = ((λ_max - λ_min)/4)^2
    τ = 2/(λ_max + λ_min)
    
    #Assign initial parameter
    β  = 2*τ
    α = 1
    b = 2/α - 1
    a = (2/τ -1) * b
    κ = τ
    CSSORIterable{T,typeof(x),typeof(x),typeof(d),typeof(ω)}(
        StrictlyLowerTriangular(A, D ),
        StrictlyUpperTriangular(A, D ),
        FastLowerTriangular(A, D),
        FastUpperTriangular(A, D),
         δ, τ, β, α, b, a, κ,
        ω, x, similar(x), similar(x), similar(x), d, maxiter
    )
end

start(s::CSSORIterable) = 1
done(s::CSSORIterable, iteration::Int) = iteration > s.maxiter

function iterate(s::CSSORIterable{T}, iteration::Int=start(s)) where {T}
    if done(s, iteration) return nothing end

 
""" tmp = c - U * x"""
    gauss_seidel_multiply!(-one(T), s.sU, s.x, one(T), s.d, s.tmp)

""" tmp = ω * inv(L) (:==inv(Mw)) * tmp + (1 - ω) * x"""
    forward_sub!(s.ω, s.L, s.tmp, one(T) - s.ω, s.x)

""" w = d - L * tmp"""
    gauss_seidel_multiply!(-one(T), s.sL, s.tmp, one(T), s.d, s.w)

""" w = ω * inv(U)(:==inv(tr(Mw)) * w + (1 - ω) * tmp"""
    backward_sub!(s.ω, s.U, s.w, one(T) - s.ω, s.tmp)
    
    s.w = s.w - s.x

    if(iteration == 1)
        sum2!(s.α * s.τ , s.w, s.α, s.x, s.U.matrix)
    else
        sum3!(s.α * s.τ , s.w, s.α , s.x, one(T)-s.α, s.prev, s.U.matrix)
    end
    s.x, s.prev = s.w, s.x
    
    
    s.β = 1 / ( (1/s.τ)  - s.β*s.δ ) 
    s.α = s.β / s.τ
    s.b = ( ( 2*s.κ*(1- s.α) ) / s.β) + 1
    s.a = ((2/s.τ) -1) + (s.b-1) * ( (1/s.τ) + (1/s.κ) -1 )
    s.κ = s.β  + ( (1 - s.α) * s.κ)
    

    nothing, iteration + 1
end

function cheby_ssor!(x::AbstractVector, A::SparseMatrixCSC, d::AbstractVector, ω::Real, λ_max::Real, λ_min::Real;maxiter::Int = 10)
    iterable = cheby_ssor_iterable(x, A, d, ω, λ_max, λ_min, maxiter = maxiter)
    for item = iterable end
    iterable.x
end
cheby_ssor(A::AbstractMatrix, d, ω::Real, λ_max::Real, λ_min::Real; kwargs...) =
    cheby_ssor!(zeros(length(d)), A, d, ω, λ_max, λ_min; kwargs...)


mutable struct CSSORSampler{T, solT, vecT, numT <: Real}
    sL::StrictlyLowerTriangular
    sU::StrictlyUpperTriangular
    L::FastLowerTriangular
    U::FastUpperTriangular
    δ::numT
    τ::numT
    β::numT
    α::numT
    b::numT
    a::numT
    κ::numT
    ω::numT
    x::solT 
    tmp::vecT
    prev::vecT
    w::vecT
    𝛎::vecT
    c::vecT
    maxiter::Int
end


function cheby_ssor_samiter(x::AbstractVector{T}, A::SparseMatrixCSC, ω::Float64, λ_max::Float64, λ_min::Float64; 𝛎::AbstractVector = [0], maxiter::Int = 10) where {T}
    D = DiagonalIndices(A)
   
    
    δ = ((λ_max - λ_min)/4)^2
    τ = 2/(λ_max + λ_min)
    
    #Assign initial parameter
    β  = 2*τ
    α = 1
    b = 2/α - 1
    a = (2/τ -1) * b
    κ = τ
    CSSORSampler{T,typeof(x),typeof(x),typeof(ω)}(
        StrictlyLowerTriangular(A, D ),
        StrictlyUpperTriangular(A, D ),
        FastLowerTriangular(A, D),
        FastUpperTriangular(A, D),
        δ, τ, β, α, b, a, κ,
        ω, x, similar(x), similar(x), similar(x), 𝛎, similar(𝛎), maxiter
    )
end

start(s::CSSORSampler) = 1
done(s::CSSORSampler, iteration::Int) = iteration > s.maxiter

function iterate(s::CSSORSampler{T}, iteration::Int=start(s)) where {T}
    if done(s, iteration) return nothing end
    rand!(Product(Normal.(s.𝛎,1)), s.c)
""" s.c= sqrt(s.b) * Dω * s.c """
    noise!(sqrt(s.b * (2/ω-1)), s.c , s.U.diag)
 
""" tmp = c - U * x
"""
    gauss_seidel_multiply!(-one(T), s.sU, s.x, one(T), s.c, s.tmp)

""" tmp = ω * inv(L) (:==inv(Mw)) * tmp + (1 - ω) * x"""
    forward_sub!(s.ω, s.L, s.tmp, one(T) - s.ω, s.x)
    
    
    rand!(Product(Normal.(s.𝛎,1)), s.c)

    noise!(sqrt(abs(s.a) * (2/ω-1)), s.c , s.U.diag)
""" w = d - L * tmp"""
    gauss_seidel_multiply!(-one(T), s.sL, s.tmp, one(T), s.c, s.w)

""" w = ω * inv(U)(:==inv(tr(Mw)) * w + (1 - ω) * tmp"""
    backward_sub!(s.ω, s.U, s.w, one(T) - s.ω, s.tmp)
    
    s.w = s.w - s.x

    if(iteration == 1)
""" x = α(x + τw)"""
        sum2!(s.α * s.τ , s.w, s.α, s.x, s.U.matrix)
    else
""" x = α(x - prev + τw) + prev """
        sum3!(s.α * s.τ , s.w, s.α , s.x, one(T)-s.α, s.prev, s.U.matrix)
    end
    s.x, s.prev = s.w, s.x
    
    
    s.β = 1 / ( (1/s.τ)  - s.β*s.δ ) 
    s.α = s.β / s.τ
    s.b = ( ( 2*s.κ*(1- s.α) ) / s.β) + 1
    s.a = ((2/s.τ) -1) + (s.b-1) * ( (1/s.τ) + (1/s.κ) -1 )
    s.κ = s.β  + ( (1 - s.α) * s.κ)
    

    nothing, iteration + 1
end

function cheby_ssor_sampler!(x::AbstractVector, A::SparseMatrixCSC,  ω::Float64, λ_max::Float64, λ_min::Float64; 𝛎::AbstractVector = zeros(1), maxiter::Int = 10)
    iterable = cheby_ssor_samiter(x, A,  ω, λ_max, λ_min, 𝛎 = 𝛎, maxiter = maxiter)
    for item = iterable end
    iterable.x
end

cheby_ssor_sampler(A::AbstractMatrix,  ω::Float64, λ_max::Float64, λ_min::Float64; kwargs...) =
    cheby_ssor_sampler!(zeros(A.n), A, ω, λ_max, λ_min; kwargs...)




"""
------------------------------------------------------------------------------
Def of calculating functions
------------------------------------------------------------------------------
"""

function eigMm(A::SparseMatrixCSC, ω::Real)

    L = (LowerTriangular(A)- (1-1/ω) * Diagonal(A))*(inv(sqrt((2/ω-1)) * Diagonal(sqrt.(diag(A)))))

    Meig =  inv(cholesky(L*L')) * A
    
    λ_max = eigs(Meig; nev=1, ritzvec=false, which=:LM)[1][1]
    λ_min = eigs(Meig; nev=1, ritzvec=false, which=:SM)[1][1]
    real(λ_max), real(λ_min)
end

function noise!(k, c, D::DiagonalIndices)
    @inbounds for i = 1 : D.matrix.n
        c[i] = k * sqrt(D.diag[i]) * c[i]
    end
    c
end

function sum2!(a, x, b, r, A::SparseMatrixCSC)
    @inbounds for i =1: A.n
        x[i] = a * x[i] + b * r[i]
    end
    x
end

function sum3!(a, x, b, r, c, z, A::SparseMatrixCSC)
    @inbounds for i =1: A.n
        x[i] = a*x[i] + b*r[i] + c * z[i]
    end
    x
end


"""
Computes z := α * U * x + β * y. Because U is StrictlyUpperTriangular
one can set z = x and update x in-place as x := α * U * x + β * y.
"""

function gauss_seidel_multiply!(α, U::StrictlyUpperTriangular, x::AbstractVector, β, y::AbstractVector, z::AbstractVector)
    A = U.matrix

    for col = 1 : A.n
        αx = α * x[col]
        diag_index = U.diag[col]
        @inbounds for j = A.colptr[col] : diag_index - 1
            z[A.rowval[j]] += A.nzval[j] * αx
        end
        z[col] = β * y[col]
    end
    z
end

"""
Computes z := α * L * x + β * y. Because A is StrictlyLowerTriangular
one can set z = x and update x in-place as x := α * L * x + β * y.
"""

function gauss_seidel_multiply!(α, L::StrictlyLowerTriangular, x::AbstractVector, β, y::AbstractVector, z::AbstractVector)
    A = L.matrix

    for col = A.n : -1 : 1
        αx = α * x[col]
        z[col] = β * y[col]
        @inbounds for j = L.diag[col] + 1 : (A.colptr[col + 1] - 1)
            z[A.rowval[j]] += A.nzval[j] * αx
        end
    end
    z
end

"""
Use like this

M, m = eigMm(A , ω)
cheby_ssor_sampler(A, b, ω, M, m,𝛎 =randn(100) , maxiter=100000)

"""