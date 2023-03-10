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
function forward_sub!(??, F::FastLowerTriangular, x::AbstractVector, ??, y::AbstractVector)
    A = F.matrix

    @inbounds for col = 1 : A.n

        # Solve for diagonal element
        idx = F.diag[col]
        x[col] = ?? * x[col] / A.nzval[idx] + ?? * y[col]

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

function backward_sub!(??, F::FastUpperTriangular, x::AbstractVector, ??, y::AbstractVector)
    A = F.matrix

    @inbounds for col = A.n : -1 : 1

        # Solve for diagonal element
        idx = F.diag[col]
        x[col] = ?? * x[col] / A.nzval[idx] + ?? * y[col]

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
    ??::numT
    x::solT
    tmp::vecT
    b::rhsT
    maxiter::Int
end

function ssor_iterable(x::AbstractVector{T}, A::SparseMatrixCSC, b::AbstractVector, ??::Real; maxiter::Int = 10) where {T}
    D = DiagonalIndices(A)
    SSORIterable{T,typeof(x),typeof(x),typeof(b),typeof(??)}(
        StrictlyLowerTriangular(A, D),
        StrictlyUpperTriangular(A, D),
        FastLowerTriangular(A, D),
        FastUpperTriangular(A, D),
        ??, x, similar(x), b, maxiter
    )
end

start(s::SSORIterable) = 1
done(s::SSORIterable, iteration::Int) = iteration > s.maxiter

function iterate(s::SSORIterable{T}, iteration::Int=start(s)) where {T}
    if done(s, iteration) return nothing end

    # tmp = b - U * x
    gauss_seidel_multiply!(-one(T), s.sU, s.x, one(T), s.b, s.tmp)

    # tmp = ?? * inv(L) * tmp + (1 - ??) * x
    forward_sub!(s.??, s.L, s.tmp, one(T) - s.??, s.x)

    # x = b - L * tmp
    gauss_seidel_multiply!(-one(T), s.sL, s.tmp, one(T), s.b, s.x)

    # x = ?? * inv(U) * x + (1 - ??) * tmp
    backward_sub!(s.??, s.U, s.x, one(T) - s.??, s.tmp)

    nothing, iteration + 1
end


"""
ssor!(x, A::SparseMatrixCSC, b, ??::Real; maxiter=10)
Performs exactly `maxiter` SSOR iterations with relaxation parameter `??`. Each iteration
is basically a forward *and* backward sweep of SOR.
Allocates a temporary vector and precomputes the diagonal indices.
Throws `LinearAlgebra.SingularException` when the diagonal has a zero. This check
is performed once beforehand.
"""

function ssor!(x::AbstractVector, A::SparseMatrixCSC, b::AbstractVector, ??::Real; maxiter::Int = 10)
    iterable = ssor_iterable(x, A, b, ??, maxiter = maxiter)
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
    ??::numT
    ??::numT
    ??::numT
    ??::numT
    b::numT
    a::numT
    ??::numT
    ??::numT
    x::solT
    tmp::vecT
    prev::vecT
    w::vecT
    d::rhsT
    maxiter::Int
end


function cheby_ssor_iterable(x::AbstractVector{T}, A::SparseMatrixCSC, d::AbstractVector, ??::Real, ??_max::Real, ??_min::Real; maxiter::Int = 10) where {T}
    D = DiagonalIndices(A)
    #Dw = sqrt((2/??-1)) * Diagonal(sqrt.(diag(A)))
    
    
    ?? = ((??_max - ??_min)/4)^2
    ?? = 2/(??_max + ??_min)
    
    #Assign initial parameter
    ??  = 2*??
    ?? = 1
    b = 2/?? - 1
    a = (2/?? -1) * b
    ?? = ??
    CSSORIterable{T,typeof(x),typeof(x),typeof(d),typeof(??)}(
        StrictlyLowerTriangular(A, D ),
        StrictlyUpperTriangular(A, D ),
        FastLowerTriangular(A, D),
        FastUpperTriangular(A, D),
         ??, ??, ??, ??, b, a, ??,
        ??, x, similar(x), similar(x), similar(x), d, maxiter
    )
end

start(s::CSSORIterable) = 1
done(s::CSSORIterable, iteration::Int) = iteration > s.maxiter

function iterate(s::CSSORIterable{T}, iteration::Int=start(s)) where {T}
    if done(s, iteration) return nothing end

 
""" tmp = c - U * x"""
    gauss_seidel_multiply!(-one(T), s.sU, s.x, one(T), s.d, s.tmp)

""" tmp = ?? * inv(L) (:==inv(Mw)) * tmp + (1 - ??) * x"""
    forward_sub!(s.??, s.L, s.tmp, one(T) - s.??, s.x)

""" w = d - L * tmp"""
    gauss_seidel_multiply!(-one(T), s.sL, s.tmp, one(T), s.d, s.w)

""" w = ?? * inv(U)(:==inv(tr(Mw)) * w + (1 - ??) * tmp"""
    backward_sub!(s.??, s.U, s.w, one(T) - s.??, s.tmp)
    
    s.w = s.w - s.x

    if(iteration == 1)
        sum2!(s.?? * s.?? , s.w, s.??, s.x, s.U.matrix)
    else
        sum3!(s.?? * s.?? , s.w, s.?? , s.x, one(T)-s.??, s.prev, s.U.matrix)
    end
    s.x, s.prev = s.w, s.x
    
    
    s.?? = 1 / ( (1/s.??)  - s.??*s.?? ) 
    s.?? = s.?? / s.??
    s.b = ( ( 2*s.??*(1- s.??) ) / s.??) + 1
    s.a = ((2/s.??) -1) + (s.b-1) * ( (1/s.??) + (1/s.??) -1 )
    s.?? = s.??  + ( (1 - s.??) * s.??)
    

    nothing, iteration + 1
end

function cheby_ssor!(x::AbstractVector, A::SparseMatrixCSC, d::AbstractVector, ??::Real, ??_max::Real, ??_min::Real;maxiter::Int = 10)
    iterable = cheby_ssor_iterable(x, A, d, ??, ??_max, ??_min, maxiter = maxiter)
    for item = iterable end
    iterable.x
end
cheby_ssor(A::AbstractMatrix, d, ??::Real, ??_max::Real, ??_min::Real; kwargs...) =
    cheby_ssor!(zeros(length(d)), A, d, ??, ??_max, ??_min; kwargs...)


mutable struct CSSORSampler{T, solT, vecT, numT <: Real}
    sL::StrictlyLowerTriangular
    sU::StrictlyUpperTriangular
    L::FastLowerTriangular
    U::FastUpperTriangular
    ??::numT
    ??::numT
    ??::numT
    ??::numT
    b::numT
    a::numT
    ??::numT
    ??::numT
    x::solT 
    tmp::vecT
    prev::vecT
    w::vecT
    ????::vecT
    c::vecT
    maxiter::Int
end


function cheby_ssor_samiter(x::AbstractVector{T}, A::SparseMatrixCSC, ??::Float64, ??_max::Float64, ??_min::Float64; ????::AbstractVector = [0], maxiter::Int = 10) where {T}
    D = DiagonalIndices(A)
   
    
    ?? = ((??_max - ??_min)/4)^2
    ?? = 2/(??_max + ??_min)
    
    #Assign initial parameter
    ??  = 2*??
    ?? = 1
    b = 2/?? - 1
    a = (2/?? -1) * b
    ?? = ??
    CSSORSampler{T,typeof(x),typeof(x),typeof(??)}(
        StrictlyLowerTriangular(A, D ),
        StrictlyUpperTriangular(A, D ),
        FastLowerTriangular(A, D),
        FastUpperTriangular(A, D),
        ??, ??, ??, ??, b, a, ??,
        ??, x, similar(x), similar(x), similar(x), ????, similar(????), maxiter
    )
end

start(s::CSSORSampler) = 1
done(s::CSSORSampler, iteration::Int) = iteration > s.maxiter

function iterate(s::CSSORSampler{T}, iteration::Int=start(s)) where {T}
    if done(s, iteration) return nothing end
    rand!(Product(Normal.(s.????,1)), s.c)
""" s.c= sqrt(s.b) * D?? * s.c """
    noise!(sqrt(s.b * (2/??-1)), s.c , s.U.diag)
 
""" tmp = c - U * x
"""
    gauss_seidel_multiply!(-one(T), s.sU, s.x, one(T), s.c, s.tmp)

""" tmp = ?? * inv(L) (:==inv(Mw)) * tmp + (1 - ??) * x"""
    forward_sub!(s.??, s.L, s.tmp, one(T) - s.??, s.x)
    
    
    rand!(Product(Normal.(s.????,1)), s.c)

    noise!(sqrt(abs(s.a) * (2/??-1)), s.c , s.U.diag)
""" w = d - L * tmp"""
    gauss_seidel_multiply!(-one(T), s.sL, s.tmp, one(T), s.c, s.w)

""" w = ?? * inv(U)(:==inv(tr(Mw)) * w + (1 - ??) * tmp"""
    backward_sub!(s.??, s.U, s.w, one(T) - s.??, s.tmp)
    
    s.w = s.w - s.x

    if(iteration == 1)
""" x = ??(x + ??w)"""
        sum2!(s.?? * s.?? , s.w, s.??, s.x, s.U.matrix)
    else
""" x = ??(x - prev + ??w) + prev """
        sum3!(s.?? * s.?? , s.w, s.?? , s.x, one(T)-s.??, s.prev, s.U.matrix)
    end
    s.x, s.prev = s.w, s.x
    
    
    s.?? = 1 / ( (1/s.??)  - s.??*s.?? ) 
    s.?? = s.?? / s.??
    s.b = ( ( 2*s.??*(1- s.??) ) / s.??) + 1
    s.a = ((2/s.??) -1) + (s.b-1) * ( (1/s.??) + (1/s.??) -1 )
    s.?? = s.??  + ( (1 - s.??) * s.??)
    

    nothing, iteration + 1
end

function cheby_ssor_sampler!(x::AbstractVector, A::SparseMatrixCSC,  ??::Float64, ??_max::Float64, ??_min::Float64; ????::AbstractVector = zeros(1), maxiter::Int = 10)
    iterable = cheby_ssor_samiter(x, A,  ??, ??_max, ??_min, ???? = ????, maxiter = maxiter)
    for item = iterable end
    iterable.x
end

cheby_ssor_sampler(A::AbstractMatrix,  ??::Float64, ??_max::Float64, ??_min::Float64; kwargs...) =
    cheby_ssor_sampler!(zeros(A.n), A, ??, ??_max, ??_min; kwargs...)




"""
------------------------------------------------------------------------------
Def of calculating functions
------------------------------------------------------------------------------
"""

function eigMm(A::SparseMatrixCSC, ??::Real)

    L = (LowerTriangular(A)- (1-1/??) * Diagonal(A))*(inv(sqrt((2/??-1)) * Diagonal(sqrt.(diag(A)))))

    Meig =  inv(cholesky(L*L')) * A
    
    ??_max = eigs(Meig; nev=1, ritzvec=false, which=:LM)[1][1]
    ??_min = eigs(Meig; nev=1, ritzvec=false, which=:SM)[1][1]
    real(??_max), real(??_min)
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
Computes z := ?? * U * x + ?? * y. Because U is StrictlyUpperTriangular
one can set z = x and update x in-place as x := ?? * U * x + ?? * y.
"""

function gauss_seidel_multiply!(??, U::StrictlyUpperTriangular, x::AbstractVector, ??, y::AbstractVector, z::AbstractVector)
    A = U.matrix

    for col = 1 : A.n
        ??x = ?? * x[col]
        diag_index = U.diag[col]
        @inbounds for j = A.colptr[col] : diag_index - 1
            z[A.rowval[j]] += A.nzval[j] * ??x
        end
        z[col] = ?? * y[col]
    end
    z
end

"""
Computes z := ?? * L * x + ?? * y. Because A is StrictlyLowerTriangular
one can set z = x and update x in-place as x := ?? * L * x + ?? * y.
"""

function gauss_seidel_multiply!(??, L::StrictlyLowerTriangular, x::AbstractVector, ??, y::AbstractVector, z::AbstractVector)
    A = L.matrix

    for col = A.n : -1 : 1
        ??x = ?? * x[col]
        z[col] = ?? * y[col]
        @inbounds for j = L.diag[col] + 1 : (A.colptr[col + 1] - 1)
            z[A.rowval[j]] += A.nzval[j] * ??x
        end
    end
    z
end

"""
Use like this

M, m = eigMm(A , ??)
cheby_ssor_sampler(A, b, ??, M, m,???? =randn(100) , maxiter=100000)

"""