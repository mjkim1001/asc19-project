import LinearAlgebra: mul!, ldiv!
import Base: getindex, iterate
using SparseArrays, Arpack, LinearAlgebra
using BenchmarkTools, IterativeSolvers, MatrixDepot, Random


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


function forward_sub!(F::FastLowerTriangular, x::AbstractVector)
    A = F.matrix
    @inbounds for col = 1 : A.n
        idx = F.diag[col]
        x[col] /= A.nzval[idx] # ok
        for i = idx + 1 : (A.colptr[col + 1] - 1) #colptr인데 lower triangular이기 때문에 해당 col의 diagonal 아래 개수가나옴.
            x[A.rowval[i]] -= A.nzval[i] * x[col] # 이 term으로 x[n] 계산할때 그이전텀들이 다 마이너스 되어서 있음. 
        end
    end
    x
end

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

function f_mul!(α::T, O::DiagonalIndices, x::AbstractVector, b::AbstractVector, r::AbstractVector  ) where {T}
    A = O.matrix
    r[:] = b
    @inbounds for col = 1 : A.n
        αx = α * x[col]
        diag_index = O.diag[col]
        for j = A.colptr[col] : A.colptr[col + 1] - 1
            r[A.rowval[j]] += A.nzval[j] * αx 
        end
    end
    r
end

function sum!(z, r, A::SparseMatrixCSC)
    @inbounds for i =1: A.n
        z[i] += r[i]
    end
end


function m_sor!(A, D::DiagonalIndices, w)
    for d_idx in D.diag 
        A.nzval[d_idx]  *= (1/w)
    end
    @inbounds for col = 1 : A.n
        for j = A.colptr[col] :  A.colptr[col + 1] - 1
            if A.rowval[j] < col 
                A.nzval[j] = 0
            end
        end
    end
    
end


function itter_sor!(F::FastLowerTriangular, D::DiagonalIndices,
                        x::AbstractVector, b::AbstractVector, max_itter)
    A = D.matrix
    T = eltype(x)
    r =zeros(A.n)
    
    for i = 1 : max_itter 
        f_mul!(-one(T), D, x, b, r) # r <- b- Ax
        
        if norm(r) < 10^(-8)
            println("sor_itter : ",i)
            break
        end 
        
        forward_sub!(F, r)# r <- M_sor\r
        sum!(x, r, A) # x <- x +  M_sor/b        
    end
    x
    
end

function k3_sor(A, b::AbstractVector, w, maxiter)
    x = zeros(A.n)
    m_sor = copy(A)
    D = DiagonalIndices(A)
    m_sor!(m_sor, D, w)
    D_ms = DiagonalIndices(m_sor)
    itter_sor!(FastLowerTriangular(m_sor ,D_ms), D, x , b, maxiter)
end

function gamma_sqrt_diag_mul!( D::DiagonalIndices, b::AbstractVector, w ,b_c)
    A = D.matrix
    for idx in D.diag 
        b[A.rowval[idx]] *=  sqrt( b_c * ((2/w) -1) * A.nzval[idx])
    end
end

function sum2!(x,y,z, A::SparseMatrixCSC)
    @inbounds for i =1: A.n
        x[i] =y[i]+z[i]
    end
end

function itter_ssor!(F::FastLowerTriangular, U::FastUpperTriangular, D::DiagonalIndices,
                        D_t::DiagonalIndices, x::AbstractVector, b::AbstractVector
                        , w,  max_itter)
    
    A = D.matrix
    A_t = D_t.matrix
    #symetric일때도 필요한지 고려 diag정의는 새로필요한거 같음
    
    T = eltype(b)
    r = zeros(A.n)
    y = zeros(A.n)
        
    for i = 1 : max_itter 
        f_mul!(-one(T), D, x, b, r) # r_1 <-  γ * D^(1/2) * b- Ay
        
        if norm(r) < 10^(-8)
            println("ssor_itter : ",i)
            break
        end 
        
        gamma_sqrt_diag_mul!(D,r,w,1)
        forward_sub!(F, r) #r_1 <- m_sor\r_1
        gamma_sqrt_diag_mul!(D,r,w,1)
        backward_sub!(U, r)
        sum!(x, r, A)

    end
    x
end



function k3_ssor(A, b::AbstractVector, w, maxiter)
    x = zeros(A.n)
    m_sor = copy(A)
    D = DiagonalIndices(A)
    D_t = DiagonalIndices(sparse(A'))
    
    m_sor!(m_sor, D, w)
    
    D_ms = DiagonalIndices(m_sor)
    m_sor_t = sparse(m_sor')
    D_ms_t = DiagonalIndices(m_sor_t)
    
    itter_ssor!(FastLowerTriangular(m_sor ,D_ms), FastUpperTriangular(m_sor_t,D_ms_t),
                    D, D_t, x , b, w, maxiter)
end

using Distributions
mutable struct Sample_arr
    y::Matrix{Float64}
end

# r<-α(M-A)x +b
function f_mul_2!(α::T, O::DiagonalIndices, F::FastLowerTriangular, x::AbstractVector, b, r::AbstractVector  ) where {T} 
    A = O.matrix
    M = F.matrix
    r[:] = b
    @inbounds for col = 1 : A.n
        αx = α * x[col]
        diag_index = O.diag[col]
        for j = A.colptr[col] : A.colptr[col + 1] - 1
            r[A.rowval[j]] += (M.nzval[j] - A.nzval[j]) * αx 
        end
    end
    r
end

function f_mul_2!(α::T, O::DiagonalIndices, U::FastUpperTriangular, x::AbstractVector, b, r::AbstractVector  ) where {T} 
    A = O.matrix
    M = U.matrix
    r[:] = b
    @inbounds for col = 1 : A.n
        αx = α * x[col]
        diag_index = O.diag[col]
        for j = A.colptr[col] : A.colptr[col + 1] - 1
            r[A.rowval[j]] += (M.nzval[j] - A.nzval[j]) * αx 
        end
    end
    r
end



function itter_ssor_sp!(F::FastLowerTriangular, U::FastUpperTriangular, D::DiagonalIndices,
                        D_t::DiagonalIndices, x::AbstractVector, w,  max_itter)
    
    A = D.matrix
    A_t = D_t.matrix   
    T = eltype(x)
    r = zeros(A.n)
    y = zeros(A.n)
        
    for i = 1 : max_itter 
        
        z =rand(Normal(0, 1), A.n)
        gamma_sqrt_diag_mul!(D,z,w,1)   # z[A.rowval[idx]] *=  sqrt( b_c * ((2/w) -1) * A.nzval[idx])
        f_mul_2!(1, D, F, x, z, y) #y <- α(M-A)x +b
        forward_sub!(F, y) #r_1 <- m_sor\r_1
        z =rand(Normal(0, 1), 100)
        
        gamma_sqrt_diag_mul!(D,z,w,1)
        f_mul_2!(1, D_t, U, y, z, x) # r<-α(M-A)x +b 맨마지막 ㅔㅂㄱ터네 넣음
        backward_sub!(U, x)
        
        #print(x)

    end
    x
end



function k3_ssor_sp(A, w, maxiter)
    x = zeros(A.n)
    m_sor = copy(A)
    D = DiagonalIndices(A)
    D_t = DiagonalIndices(sparse(A'))
    
    m_sor!(m_sor, D, w)
    
    D_ms = DiagonalIndices(m_sor)
    m_sor_t = sparse(m_sor')
    D_ms_t = DiagonalIndices(m_sor_t)
    
    itter_ssor_sp!(FastLowerTriangular(m_sor ,D_ms), FastUpperTriangular(m_sor_t,D_ms_t),
                    D, D_t, x , w, maxiter)
end