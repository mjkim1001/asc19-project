"""
-----------------------------------------------------------------
julia itteration solver sparse src
-----------------------------------------------------------------
"""

include("sparseStructure.jl")

"""
-----------------------------------------------------------------
Custum Operation for sparse multiplication and division
-----------------------------------------------------------------
"""

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


function gamma_sqrt_diag_mul!( D::DiagonalIndices, b::AbstractVector, w ,b_c)
    A = D.matrix
    for idx in D.diag 
        b[A.rowval[idx]] *=  sqrt( b_c * ((2/w) -1) * A.nzval[idx])
    end
end


function sum!(z, r, A::SparseMatrixCSC)
    @inbounds for i =1: A.n
        z[i] += r[i]
    end
end

function sum2!(x,y,z, A::SparseMatrixCSC)
    @inbounds for i =1: A.n
        x[i] =y[i]+z[i]
    end
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

function mul_sum!(x_next, x, x_pre, r, α, τ, A)
    for i = 1:A.n 
        x_next[i] = (1-α)*x_pre[i] + α*x[i] + (τ*α*r[i])
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

function mul_inv_d!(D, r)
    A = D.matrix
    for idx in D.diag 
        r[A.rowval[idx]] *=  (1 / A.nzval[idx])
    end
end