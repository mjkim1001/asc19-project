using SparseArrays, LinearAlgebra

function rowidx(n)
    #define a basic units which consist a columns of A matrix.
    a = [0; 1; n]  #[2.0001; -1; -1]
    ac = [-1; 0; n]  #[-1; 2.0001; -1]
    b = [-1; 0; 1; n]  #for upper tail [-1; 3.0001; -1; -1]
    bf = [-n; 0; 1; n] # for center [-1; 3.0001; -1; -1]
    bf2 = [-n; -1; 0; n] 
    c = [-n; -1; 0; 1; n]  #[-1; -1; 4.0001; -1; -1]
    
    #part A
    #A = vcat(a, repeat(b, n-2), ac, b)
    
    A = a .+ 1  #i = 1
    for i in 2:n-1
        A = vcat(A, b .+ i)
    end
    A = vcat(A, ac.+n) # i = n
    A = vcat(A, bf.+(n+1))
    
    # for an index n+2 ~ n^2-n-1
    # define a Bunit
    Bunit = copy(c)
    for j in 1:n-3
        Bunit = vcat(Bunit, c.+j)
    end
    Bunit = vcat(Bunit, bf2.+(n-2))
    Bunit = vcat(Bunit, bf.+(n-1))
    
    for i in n+2:n:(n-1)^2
        A = vcat(A, Bunit.+i)
    end
    A = vcat(A, Bunit[1:end-8].+(n^2-2*n+2))
    temp = (n^2+1) .- A[1:4*n+2]
    A = vcat(A, reverse(temp))
    A
end

function nz_vals(n)
    """
    Output: non-zero values of matrix A in column-wise ordering.
    To define a SparseCSC structure.
    """
    
    #define a basic units which consist a columns of A matrix.
    a = [2.0001; -1; -1]
    at = [-1; -1; 2.0001]
    ac = [-1; 2.0001; -1]
    b = [-1; 3.0001; -1; -1]
    bt = [-1; -1; 3.0001; -1]
    c = [-1; -1; 4.0001; -1; -1]
    
    #part A
    A = vcat(a, repeat(b, n-2), ac, b)
    
    #part B
    Bunit = vcat(repeat(c, n-2), bt, b)
    B = vcat(repeat(Bunit, n-3), repeat(c, n-2))
    
    vcat(A, B, reverse(A))
end

function colA(n)
    #part A
    A = vcat(3, repeat([4], n-2), 3, 4)
    #part B
    Bunit = vcat(repeat([5], n-2), 4, 4)
    B = vcat(repeat(Bunit, n-3), repeat([5], n-2))
    
    # calculate the colptr
    colnzs = vcat(A, B, reverse(A))
    colptr = ones(Int64, n^2+1)
    for i in 1:n^2
        colptr[i+1] = colptr[i] + colnzs[i]
    end
    colptr
end

function laplacematrix(n)
    SparseMatrixCSC(n^2, n^2, colA(n), rowidx(n), nz_vals(n))
end