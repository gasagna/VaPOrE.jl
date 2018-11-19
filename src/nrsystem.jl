# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

# function to calculate the negative redidual
function calc_neg_residual!(r::PeriodicOrbit{T, X},
                            F,
                            order::Int,
                            q::PeriodicOrbit{T, X},
                            tmp1::X, tmp2::X) where {T, X}
    # make sure we do not do shit!
    @checkorder order
    for i = 1:length(q)
        r[i] .= .- q.ω.*dds!(q, i, tmp1, order) .+ F(0.0, q[i], tmp2)
    end
    return r
end

# global row indices of the i-th block (one-based)
@inline _blockrng(i::Integer, N::Integer) = ((i-1)*N+1):(i*N)

struct Cache{T, X, FT, LT, DT, RT}
    A::SparseMatrixCSC{T, Int}
    b::Vector{T}
    tmp::X
    TMP::Matrix{T}
    diffmat::Matrix{T}
    F::FT
    L::LT
    D::DT
    R::RT
    order::Int
    dq_newton::X
    dq_cauchy::X
    function Cache(q::PeriodicOrbit{T, X}, order::Int, F, L, D, R) where {T, X}
        # check order
        @checkorder order

        # get problem size
        M, N = length(q), length(q[1])

        # preallocate the diagonals, to speed up filling
        c = div(order, 2)
        
        idxs = tuple(-c*N:c*N...)
        vecs = ntuple(i->zeros(T, M*N-abs(idxs[i])), length(idxs))
        # if we pass nothing the the constructor for D, we have no shift
        d = D isa Void ? 1 : 2
        A = spdiagm(vecs, idxs, M*N+d, M*N+d)

        # precompute differentiation matrix from the operator D
        diffmat = zeros(T, N, N)
        if !(D isa Void)
            tmp1 = similar(q[1])
            tmp2 = similar(q[1])
            tmp1 .= 0
            for i = 1:N
                tmp1[i] = 1
                D(tmp2, tmp1)
                diffmat[:, i] .= tmp2
                tmp1[i] = 0
            end
        end

        # instantiate
        new{T, X, typeof(F), typeof(L), typeof(D), typeof(R)}(
            A, zeros(T, M*N+d),
            similar(q[1]),
            zeros(T, N, N), diffmat, F, L, D, R, order)
    end
end

function update!(c::Cache, q::PeriodicOrbit)
    # get problem size
    M, N = length(q), length(q[1])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CALCULATE THE NEGATIVE RESIDUAL
    calc_neg_r(c.neg_r, c.F, c.order, q, c.tmp1, c.tmp2)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # UDPATE LHS OF NEWTON SYSTEM
    @inbounds begin

        # time derivative term
        ds = 2π/length(q)
        for (j, c) in enumerate(_coeffs[sys.order])
            c.A[diagind(c.A,      j*N)] .= +c .* q.ω ./ ds
            c.A[diagind(c.A,     -j*N)] .= -c .* q.ω ./ ds
            c.A[diagind(c.A,  (M-j)*N)] .= -c .* q.ω ./ ds
            c.A[diagind(c.A, -(M-j)*N)] .= +c .* q.ω ./ ds
        end

        # clear last rows/cols
        c.A[end, :] .= 0
        c.A[:, end] .= 0
        
        # diagonal block
        for i in 1:M
            # diagonal blocks
            c.A[_blockrng(i, N), _blockrng(i, N)] .= .-c.L(c.TMP, q[i])

            if !(c.D isa Void)
                # diagonal term associate to v
                c.A[_blockrng(i, N), _blockrng(i, N)] .+= q.v .* c.diffmat

                # right column term associated to v'
                c.A[_blockrng(i, N), end-1] .= c.D(c.tmp, q[i])
            end

            # term associated to ω'
            c.A[_blockrng(i, N),  end] .= dds!(q, i, c.tmp, c.order)
        end

        # now add phase locking constraints
        c.D isa Void || (c.A[end-1, _blockrng(1, N)] = c.D(c.tmp, q[1]))
        c.A[end, _blockrng(1, N)] .= c.F(0.0, q[1], c.tmp)
    end

    return nothing
end

# Solve newton step
function solve_newton!(c::Cache)
    # get problem size
    M, N = length(c.dq_newton), length(c.dq_newton[1])

    # set negative residual into the b vector
    for i in 1:M
        c.b[_blockrng(i, N)] .= c.neg_r[i]
    end
    
    # set last bits to zero
    c.D isa Void || (c.b[end-1] = 0)
    c.b[end] = 0

    # solve Newton system
    A_ldiv_B!(lufact(c.A), c.b)

    # now copy to dq_newton
    for i in 1:length(c.dq_newton)
        c.dq_newton[i] .= c.b[_blockrng(i, N)]
    end
    
    # and the other bits too
    c.D isa Void || (c.dq_newton.v = c.b[end-1])
    c.dq_newton.ω = c.b[end]

    return nothing
end


function solve_cauchy!(cache::Cache)
    # calc residual
    cache.r 

    mul!(cache.grad, cache.G_adj, cache.res)

    # now compute denominator bit
    mul!(cache.tmp, cache.G, cache.grad)

    # this is the cauchy step length
    λᵒ = (norm(cache.grad)/norm(cache.tmp))^2

    # obtain the cauchy length
    cache.dq_cauchy .= λᵒ .* cache.grad

    return nothing
end