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
    #x
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

    #x
end