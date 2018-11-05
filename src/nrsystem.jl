# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

export NRSystem, solve!, update!

# global row indices of the i-th block (one-based)
@inline _blockrng(i::Integer, N::Integer) = ((i-1)*N+1):(i*N)

struct NRSystem{T, X, FT, LT, DT, RT}
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
    function NRSystem(x::PeriodicOrbit{T, X}, order::Int, F, L, D, R) where {T, X}
        # check order
        @checkorder order

        # get problem size
        M, N = length(x), length(x[1])

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
            tmp1 = similar(x[1])
            tmp2 = similar(x[1])
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
            similar(x[1]),
            zeros(T, N, N), diffmat, F, L, D, R, order)
    end
end

# update Newton-Raphson matrix with current guess
function update!(sys::NRSystem, x::PeriodicOrbit)
    # get problem size
    M, N = length(x), length(x[1])

    # UDPATE LHS
    @inbounds begin

        # time derivative term
        ds = 2π/length(x)
        for (j, c) in enumerate(_coeffs[sys.order])
            sys.A[diagind(sys.A,      j*N)] .= +c .* x.ω ./ ds
            sys.A[diagind(sys.A,     -j*N)] .= -c .* x.ω ./ ds
            sys.A[diagind(sys.A,  (M-j)*N)] .= -c .* x.ω ./ ds
            sys.A[diagind(sys.A, -(M-j)*N)] .= +c .* x.ω ./ ds
        end

        # clear last rows/cols
        sys.A[end, :] .= 0
        sys.A[:, end] .= 0
        
        # diagonal block
        for i in 1:M
            # diagonal blocks
            sys.A[_blockrng(i, N), _blockrng(i, N)] .= .-sys.L(sys.TMP, x[i])

            if !(sys.D isa Void)
                # diagonal term associate to v
                sys.A[_blockrng(i, N), _blockrng(i, N)] .+= x.v .* sys.diffmat

                # right column term associated to v'
                sys.A[_blockrng(i, N), end-1] .= sys.D(sys.tmp, x[i])
            end

            # term associated to ω'
            sys.A[_blockrng(i, N),  end] .= dds!(x, i, sys.tmp, sys.order)
        end

        # now add phase locking constraints
        sys.D isa Void || (sys.A[end-1, _blockrng(1, N)] = sys.D(sys.tmp, x[1]))
        sys.A[end, _blockrng(1, N)] .= sys.F(0.0, x[1], sys.tmp)

        # UPDATE RHS with negative residual
        for i in 1:M
            sys.b[_blockrng(i, N)] .= sys.R(sys.tmp, x, i)
        end
        
        # set last bits to zero
        sys.D isa Void || (sys.b[end-1] = 0)
        sys.b[end] = 0
    end

    return sys
end

function solve!(sys::NRSystem, x::PeriodicOrbit)
    # solve system in place
    A_ldiv_B!(lufact(sys.A), sys.b)

    # get problem size
    M, N = length(x), length(x[1])

    # now copy to x
    for i in 1:M
        x[i] .= sys.b[_blockrng(i, N)]
    end
    
    # and the other bits
    sys.D isa Void || (x.v = sys.b[end-1])
    x.ω = sys.b[end]

    return x
end
