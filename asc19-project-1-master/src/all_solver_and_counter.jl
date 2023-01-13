"""
-----------------------------------------------------------------
julia itteration solver sparse src
-----------------------------------------------------------------
"""

include("sparseStructure.jl")
include("sparseOperation.jl")
# K3 src

"""
-----------------------------------------------------------------
SOR
-----------------------------------------------------------------
"""

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


"""
-----------------------------------------------------------------
SSOR
-----------------------------------------------------------------
"""
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



# ssor sampler

function itter_ssor_sp!(F::FastLowerTriangular, U::FastUpperTriangular, D::DiagonalIndices,
                        D_t::DiagonalIndices, x::AbstractVector, w,  max_itter)
    
    A = D.matrix
    A_t = D_t.matrix   
    T = eltype(x)
    r = zeros(A.n)
    y = zeros(A.n)
        
    for i = 1 : max_itter 
        
        z = rand(Normal(0, 1), A.n)
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



"""
-----------------------------------------------------------------
Cheby_SSOR
-----------------------------------------------------------------
"""
mutable struct CB_variable
    β::Float64
    α::Float64
    b::Float64
    a::Float64
    κ::Float64
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




function itter_CB_ssor_sp!(F::FastLowerTriangular, U::FastUpperTriangular, D::DiagonalIndices,
                        D_t::DiagonalIndices, x::AbstractVector, ν, w,  λ_max, λ_min, max_itter)
    
    A = D.matrix
    A_t = D_t.matrix
    
    δ = ((λ_max - λ_min)/4)^2
    τ = 2/(λ_max + λ_min)
    #print("hi")

    T = eltype(x)
    cb = CB_variable(0,0,0,0,0)
    #Assign initial parameter
    cb.β  = 2*τ
    cb.α = 1
    cb.b = 2/cb.α - 1
    cb.a = (2/τ -1) * cb.b
    cb.κ = τ
    
    T = eltype(x)
    r_1 = zeros(A.n)
    r_2 = zeros(A.n)
    x_pre = zeros(A.n)
    x_next = zeros(A.n)
    x_temp = zeros(A.n)
    w_v = zeros(A.n)

    d = MvNormal( ν , ones( A.n))
    
    for i = 1 : max_itter 

        x_pre[:] = x 
        x[:] = x_next
        
        z = rand(d) 
        gamma_sqrt_diag_mul!( D, z, w , cb.b) #z <- z *  b^1/2 D_w^1/2 

        f_mul!(-one(T), D, x, z, r_1) # r <- z - A* X                
        forward_sub!(F, r_1) #r_1 <- m_sor\r_1
        sum2!(x_temp, x, r_1, A) #x_next <- x + τ*r
        
        z = rand(d) 
        gamma_sqrt_diag_mul!( D, z, w , cb.a) #z <- z *  b^1/2 D_w^1/2         
        
        f_mul!(-one(T), D, x_temp, z, r_2) # r_2 <- z - A* X
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

function k3_CB_ssor_sp(A, ν, w, λ_max, λ_min,  maxiter)
    x = zeros(A.n)
    m_sor = copy(A)
    D = DiagonalIndices(A)
    D_t = DiagonalIndices(sparse(A'))
    m_sor!(m_sor, D, w)

    D_ms = DiagonalIndices(m_sor)
    m_sor_t = sparse(m_sor')
    D_ms_t = DiagonalIndices(m_sor_t)

    itter_CB_ssor_sp!(FastLowerTriangular(m_sor ,D_ms), FastUpperTriangular(m_sor_t,D_ms_t),
                    D, D_t, x , ν, w, λ_max, λ_min, maxiter)
end


"""
-----------------------------------------------------------------
Richardson
-----------------------------------------------------------------
"""
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


"""
-----------------------------------------------------------------
Jacobi
-----------------------------------------------------------------
"""
    
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


"""
-----------------------------------------------------------------
Gauss-Seider
-----------------------------------------------------------------
"""
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
            println("HIHI : ",i)
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
