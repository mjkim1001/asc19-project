include("sparseStructure.jl")

"""
Algorithm 1: solve Ax=b by cholesky factorization
# solve Ax = b by cholesky factorization
# LL'x = b
# L'x = forward(L, b)
"""

function choleskysolver!(A, b)
    L = sparse(cholesky(A).L)
    D = DiagonalIndices(L)
    fastL = FastLowerTriangular(L, D)
    
    Lt = sparse(Matrix(L'))
    Dt = DiagonalIndices(Lt)
    fastU = FastUpperTriangular(Lt, Dt)
    
    forward_sub!(fastL, b)
    backward_sub!(fastU, b)
end


"""
Algorithm 2: Cholesky sampling using a precision matrix A

input: Precision matrix A
output: y ~ N(0. inv(A))

psuedo algorithm
1. Cholesky factor A = BB';
2. sample z ~ N(0, I);
3. solve B'y = z by back subsititution;
"""

function choleskySampler!(A)
    # input matrix A is a SparseMatrixCSC{Float64,Int64}.
    A[:] = sparse(cholesky(A).L)' # cholesky factorization with in-place
    U = FastUpperTriangular(A, DiagonalIndices(A))
    z = rand(MvNormal(zeros(A.n), I))
    backward_sub!(U, z)
    z
end

function choleskySampler(A)
    choleskySampler!(copy(A))
end