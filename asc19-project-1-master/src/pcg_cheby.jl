#######################################
# PCG - Chebyshev Accelerated Sampler #
######################################
####
#combined PCG sampling - Eigenvalue finding - Chebyshev sampling
###
"""
  y ~ N(A^(-1)b , A^(-1))
  x ~ A^(-1)b
"""
include("../etc/cheby_sampler_basedonJulia.jl")

function pcg_cheby!(x, A, y0, b;
    ω::Real = 1,
    tol = sqrt(eps(real(eltype(b)))),
    maxiter_pcg::Int = size(A, 2),
    maxiter_cheby::Int = 10,
    Pl = Identity()
)
    pcg_samp = cgSamp!(x, A, y0, b;
        tol = tol,
        maxiter = maxiter_pcg,
        Pl = Pl
    )
    print("pcg_end\n")
    Lz = lanczos(pcg_samp.a, pcg_samp.b)
    M = eigs(Matrix(Lz); nev=1, ritzvec=false, which=:LM)[1][1] 
    m = eigs(Matrix(Lz); nev=1, ritzvec=false, which=:SM)[1][1]
    print("cheby_start\n")
    cheby = cheby_ssor_sampler!(pcg_samp.y, A, ω, real(M), real(m); 𝛎 = zeros(A.n), maxiter = maxiter_cheby)
    
    return (cheby + pcg_samp.x, pcg_samp.x)
end



"""
implement lanzcos matrix
"""


function nz_vals(k, a, b)
    # a: k-vector, b: k-1 vector
    nzs = vcat(a[1], b[1])
    ck = vcat(b[k-1], a[k])
    for i in 2:k-1
        colunit = vcat(b[i-1], a[i], b[i])
        nzs = vcat(nzs, colunit)
    end
    nzs = vcat(nzs, ck)
    nzs
end

function rowidx(k)
    idx = [1; 2]
    unit = [-1; 0; 1]
    for i in 2:k-1
        idx = vcat(idx, (unit .+ i))
    end
    idx = vcat(idx, [k-1; k])
    idx
end

function colptr(k)
    seq = vcat(2, repeat([3], k-2), 2)
    temp = ones(Int64, k+1)
    for i in 2:k+1
        temp[i] = temp[i-1] + seq[i-1]
    end
    temp
end

function lanczos(a, b)
    k = length(a)
    cptr = colptr(k)
    ridx = rowidx(k)
    """
    while(length(ridx) != (cptr[end]-1))
        ridx = rowidx(k)
    end
    """
    print(ridx)
    nz = nz_vals(k,a,b)
    SparseMatrixCSC(k, k, cptr, ridx, nz)
end