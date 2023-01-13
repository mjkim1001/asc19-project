"""
Generating A matrix in Sec 6.1 of Fox and Parker(Bernoulli, 2017).

"""

using LinearAlgebra, SparseArrays

function cartesianidx(k::Int, n::Int)
    """
    {s_i} are on a regular n by n lattice over the two dimensional domain
    S = [1, n] \times [1, n].
    output gives a cartesian index (i, j) of s_k. 
    For example, n = 10 gives
    S = [s01 s11 s21 ... s91
         s02 s12 s22 ... s92
         s03 s13 s23 ... s93
            ... ... 
         s10 s20 s30 ... s100].
    """
    if k % n == 0
        [n; k ÷ n]
    else
        [k % n; k ÷ n + 1]
    end
end


"""
gen_Ab(n) generate n^2 by n^2 unscaled Laplacian prior precision matrix
considered by Higdon(2006), Rue and Held(2005).
"""
function gen_A(n)
    # n_i is the number of points neighbouring s_i, i.e., with distance 1from s_i
    # N[i, j] = n_k, where cartesianindex(k, n) = (i, j)
    N = 4 * ones(n, n);
    for k in 1:n
        N[1, k] -= 1
        N[k, 1] -= 1
        N[n, k] -= 1
        N[k, n] -= 1
    end
    A = zeros(n^2, n^2)
    for i in 1:n^2
        for j in 1:n^2
            if i == j
                A[i, j] = 0.0001 + N[i]
            elseif norm(cartesianidx(i, n)-cartesianidx(j, n)) <= 1.0
                A[i, j] = -1.0
            else
                A[i, j] = 0.0
            end
        end
    end
    A = sparse(A)
    A
end