# Author: Dongwook Kim
# Rearrangement: Minwoo Kim

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



# SSOR

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


# Cheby_SSOR

mutable struct CB_variable
    β::Float64
    α::Float64
    b::Float64
    a::Float64
    κ::Float64
end

function mul_sum!(x_next, x, x_pre, r, α, τ, A)
    for i = 1:A.n 
        x_next[i] = (1-α)*x_pre[i] + α*x[i] + (τ*α*r[i])
    end
    # x
end

function sum2!(x_next, x, r, A)
    for i = 1:A.n 
        x_next[i] = x[i] + r[i]
    end
    # x
end

function sum3!(w_v, x_temp , x, r, A)
    for i = 1:A.n 
        w_v[i] = x_temp[i] - x[i] + r[i]
    end
    # x
end

function eigMm(A::SparseMatrixCSC, ω::Real)
    Dw = sqrt((2/ω-1)) * Diagonal(sqrt.(diag(A)))
    L = (LowerTriangular(A)- (1-1/ω) * Diagonal(A))*(inv(Dw))

    Meig = inv(cholesky(L*L')) * A
    
    λ_max = eigs(Meig; nev=1, ritzvec=false, which=:LM)[1][1]
    λ_min = eigs(Meig; nev=1, ritzvec=false, which=:SM)[1][1]
    real(λ_max), real(λ_min)
end

function itter_CB_ssor!(F::FastLowerTriangular, U::FastUpperTriangular, D::DiagonalIndices,
                        D_t::DiagonalIndices, x::AbstractVector, b::AbstractVector
                        , w,  λ_max, λ_min, max_itter)
    
    A = D.matrix
    A_t = D_t.matrix
    
    δ = ((λ_max - λ_min)/4)^2
    τ = 2/(λ_max + λ_min)
    
    T = eltype(b)
    cb = CB_variable(0,0,0,0,0)
    #Assign initial parameter
    cb.β  = 2*τ
    cb.α = 1
    cb.b = 2/cb.α - 1
    cb.a = (2/τ -1) * cb.b
    cb.κ = τ
    
    T = eltype(b)
    r_1 = zeros(A.n)
    r_2 = zeros(A.n)
    x_pre = zeros(A.n)
    x_next = zeros(A.n)
    x_temp = zeros(A.n)
    w_v = zeros(A.n)
 
    for i = 1 : max_itter 
        x_pre[:] = x 
        x[:] = x_next
        
        f_mul!(-one(T), D, x, b, r_1) # r <- b - A* X        
        if norm(r_1) < 10^(-8)
            println("CB ssor_itter : ",i)
            break
        end 
        
        forward_sub!(F, r_1) #r_1 <- m_sor\r_1
        sum2!(x_temp, x, r_1, A) #x_next <- x + τ*r
        
        f_mul!(-one(T), D, x_temp, b, r_2) # r_2 <- b - A* X
        backward_sub!(U, r_2)
        sum3!(w_v, x_temp , x, r_2, A)
        
        if i == 1
            sum2!(x_next, cb.α*x, τ*w_v, A) #x_next <- x + τ*r
        else
            mul_sum!(x_next, x, x_pre, w_v, cb.α, τ, A) # x_next <- (1-α)*x_pre + α*x + (τ*α*r[i])
        end
        
        cb.β = 1 / ( (1/τ)  - cb.β*δ ) 
        cb.α = cb.β / τ
        cb.b = ( ( 2*cb.κ*(1- cb.α) ) / cb.β) + 1
        cb.a = ((2/τ) -1) + (cb.b-1) * ( (1/τ) + (1/cb.κ) -1 )
        cb.κ = cb.β  + ( (1 - cb.α) * cb.κ)
    end
    x
end

function k3_CB_ssor(A, b::AbstractVector, w, λ_max, λ_min,  maxiter)
    x = zeros(A.n)
    m_sor = copy(A)
    D = DiagonalIndices(A)
    D_t = DiagonalIndices(sparse(A'))
    
    m_sor!(m_sor, D, w)
    
    D_ms = DiagonalIndices(m_sor)
    m_sor_t = sparse(m_sor')
    D_ms_t = DiagonalIndices(m_sor_t)
    #λ_max,λ_min = eigMm(A, w)
    
    itter_CB_ssor!(FastLowerTriangular(m_sor ,D_ms), FastUpperTriangular(m_sor_t,D_ms_t),
                    D, D_t, x , b, w, λ_max, λ_min, maxiter)
end

# Richardson

function itter_Richardson!(D::DiagonalIndices, x::AbstractVector, w, b::AbstractVector, max_itter)
    A = D.matrix
    T = eltype(x)
    r =zeros(A.n)
    
    for i = 1 : max_itter 
        f_mul!(-one(T), D, x, b, r) # r <- b- Ax
        
        if norm(r) < 10^(-8)
            println("Richardson_itter : ",i)
            break
        end 
        sum!(x, w*r, A) # x <- x +  wr
    end
    x
end

function k3_RCS(A, b::AbstractVector, w, maxiter)
    x = zeros(A.n)
    D = DiagonalIndices(A)
    itter_Richardson!(D, x , w, b, maxiter)
end

# Jacobi

function mul_inv_d!(D, r)
    A = D.matrix
    for idx in D.diag 
        r[A.rowval[idx]] *=  (1 / A.nzval[idx])
    end
end
    
function itter_JCB!(D::DiagonalIndices, x::AbstractVector, b::AbstractVector, max_itter)
    A = D.matrix
    T = eltype(x)
    r =zeros(A.n)
    
    for i = 1 : max_itter 
        f_mul!(-one(T), D, x, b, r) # r <- b- Ax
        
        if norm(r) < 10^(-8)
            println("Jacobi_itter : ",i)
            break
        end 

        mul_inv_d!(D, r)# r <- D^-1*r
        sum!(x, r, A) # x <- x +  M^-1*r
    end
    x
    
end

function k3_JCB(A, b::AbstractVector, maxiter)
    x = zeros(A.n)
    D = DiagonalIndices(A)
    itter_JCB!(D, x , b, maxiter)
end

# Gauss-Seidal

function m_GS!(A, D::DiagonalIndices)
    @inbounds for col = 1 : A.n
        for j = A.colptr[col] :  A.colptr[col + 1] - 1
            if A.rowval[j] < col 
                A.nzval[j] = 0
            end
        end
    end
    
end

function itter_GS!(F::FastLowerTriangular, D::DiagonalIndices,
                        x::AbstractVector, b::AbstractVector, max_itter)
    A = D.matrix
    T = eltype(x)
    r =zeros(A.n)
    
    for i = 1 : max_itter 
        f_mul!(-one(T), D, x, b, r) # r <- b- Ax
        if norm(r) < 10^(-8)
            println("GaussSeidel : ",i)
            break
        end 
        forward_sub!(F, r)# r <- M_sor\r
        sum!(x, r, A) # x <- x +  M_sor/b        
    end
    x
end


function k3_GS(A, b::AbstractVector, maxiter)
    x = zeros(A.n)
    m_GS = copy(A)
    D = DiagonalIndices(A)
    m_GS!(m_GS, D)
    D_GS = DiagonalIndices(m_GS)
    itter_GS!(FastLowerTriangular(m_GS ,D_GS), D, x , b, maxiter)
end