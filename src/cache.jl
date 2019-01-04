# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UTILITIES

# global row indices of the i-th block (one-based)
@inline _blockrng(i::Integer, N::Integer) = ((i-1)*N+1):(i*N)

# Apply operator op{u}*v to every column v of the identity matrix 
function op_apply_eye!(out::Matrix{T},
                       op,
                       u::X, v::X, tmp::X) where {T, X<:AbstractVector{T}}
    N = length(v)
    N == length(u) == length(tmp) == size(out, 1) == size(out, 2) ||
        throw(DimensionMismatch("invalid input"))
    v .= zero(T)
    @inbounds for i = 1:N
        v[i] = 1
        out[:, i] .= op(0.0, u, v, tmp)
        v[i] = 0
    end
    return out
end

# Single argument, when op does accept only one argument (e.g. for the derivative)
op_apply_eye!(out::Matrix{T}, op, tmp::X) where {T, X<:AbstractVector{T}} =
    op_apply_eye!(out, args->op(args[2]), nothing, tmp)

# In case we pass nothing as op, we do nothing
op_apply_eye!(out::Matrix{T}, op::Void, tmp::X) where {T, X<:AbstractVector{T}} = out


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main type
struct Cache{T, X, FT, OPT, U<:StateSpaceLoop, H<:PeriodicOrbit}
            A::SparseMatrixCSC{T, Int} # sparse matrix for newton iterations
            b::Vector{T}               # right hand side for newton iterations
          TMP::Matrix{T}               # temporary matrix to build the lhs
         dmat::Matrix{T}               # differentitation matrix
          tmp::NTuple{4, X}            # temporaries
            F::FT                      # the vector field
           op::OPT                     # operator for the cauchy step
          res::U                       # the residual
      res_tmp::U                       # a temporary
          den::U                       # term appearing at the denominator for the cauchy step
        q_tmp::H                       # temporary
    dq_newton::H                       # the newton step
    dq_cauchy::H                       # the cauchy step
         grad::H                       # gradient of mean square residual
    function Cache(q::H, F, L, L⁺, Dx) where {M, ORDER, T, X, NS,
                             U<:StateSpaceLoop{M, ORDER, T, X},
                             H<:PeriodicOrbit{U, NS}}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Problem parameters
        N = length(q.u[1])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define temporaries
        tmp = ntuple(i->similar(q.u[1]), 4)
        TMP = zeros(T, N, N)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build operator for Cauchy step calculations
        op = _Operator(L, L⁺, Dx, q)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Preallocate diagonals of the sparse matrix to speed up filling
        idxs = tuple(-div(ORDER, 2)*N:div(ORDER, 2)*N...)
        vecs = ntuple(i->zeros(T, M*N-abs(idxs[i])), length(idxs))
        A    = spdiagm(vecs, idxs, M*N+NS, M*N+NS)
        b    = zeros(T, size(A, 1))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Precompute differentiation matrix from the operator Dx.
        # In case Dx is nothing, this is just a no-op
        dmat = op_apply_eye!(zeros(T, N, N), Dx, tmp[1])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # define other fields
        q_tmp     = similar(q)
        dq_newton = similar(q)
        dq_cauchy = similar(q)
             grad = similar(q)
              res = similar(q.u)
          res_tmp = similar(q.u)
              den = similar(q.u)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # instantiate
        new{T, X, typeof(F), typeof(op), U, H}(A, b, TMP, dmat, tmp,
                                    F, op, res, res_tmp, den, q_tmp, dq_newton, dq_cauchy, grad)
    end
end

function update!(q::PeriodicOrbit, c::Cache)
    # get problem size
    M, N = length(q.u), length(q.u[1])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # UDPATE LHS OF NEWTON SYSTEM
    @inbounds begin
        # time derivative term
        ds = 2π/length(q.u)
        for (j, coeff) in enumerate(_FDCOEFFS[order(q.u)])
            v = coeff * q.ds[1] / ds
            c.A[diagind(c.A,      j*N)] .= + v
            c.A[diagind(c.A,     -j*N)] .= - v
            c.A[diagind(c.A,  (M-j)*N)] .= - v
            c.A[diagind(c.A, -(M-j)*N)] .= + v
        end

        # clear last rows/cols
        c.A[end, :] .= 0
        c.A[:, end] .= 0
        
        for i in 1:M
            rng = _blockrng(i, N)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Fill rhs
            c.A[rng, rng] .= .- op_apply_eye!(c.TMP, c.op.D, q.u[i], c.tmp[1], c.tmp[2])

            # column on the right (term associated to ω')
            c.A[rng,  end] .= dds!(q.u, i, c.tmp[1])

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set right hand side to negative residual into the b vector
            c.b[rng] .= (.-q.ds[1].*dds!(q.u, i, c.tmp[1])
                                     .+ c.F(0.0, q.u[i], c.tmp[2]))
        end

        c.b[end] = 0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Include phase locking constraints
        c.A[end, _blockrng(1, N)] .= c.F(0.0, q.u[1], c.tmp[1])
    end

    return nothing
end

# Solve newton step
function solve_newton!(c::Cache)
    # get problem size
    M, N = length(c.dq_newton.u), length(c.dq_newton.u[1])
  
    # solve Newton system in place
    A_ldiv_B!(lufact(c.A), c.b)

    # now copy to dq_newton
    for i in 1:M
        c.dq_newton.u[i] .= c.b[_blockrng(i, N)]
    end
    
    # update angular frequency and (potentially shifts)
    c.dq_newton.ds = tuple(c.b[end-length(c.dq_newton.ds)+1:end]...)

    return nothing
end


function solve_cauchy!(q::PeriodicOrbit, c::Cache)
    # update residual
    compute_residual!(c, q)

    # Get gradient of mean square residual. This
    # goes at the numerator
    At_mul_B!(c.grad, c.op, c.res)

    # now compute denominator bit
    A_mul_B!(c.den, c.op, c.grad)

    # cauchy step length (along the negative gradient)
    λ_c = (norm(c.grad)/norm(c.den))^2

    # obtain the cauchy length
    c.dq_cauchy .= .- λ_c .* c.grad
    
    return nothing
end

# Compute the residual vector at 'q'
function compute_residual!(c::Cache,
                           q::PeriodicOrbit{U}, 
                           r::StateSpaceLoop{M}) where {M, U<:StateSpaceLoop{M}}
    for i = 1:M
        r[i] .=  (q.ds[1].*dds!(q.u, i, c.tmp[1]) .- c.F(0.0, q.u[i], c.tmp[2]))
    end
    return r
end

compute_residual!(c::Cache, q::PeriodicOrbit) = compute_residual!(c, q, c.res)

# Compute the residual vector at 'q + dq'
function compute_residual!(c::Cache,
                           q::PeriodicOrbit{U},
                          dq::PeriodicOrbit{U},
                           r::StateSpaceLoop{M},
                           α::Real=1) where {M, U<:StateSpaceLoop{M}}
    for i = 1:M
        c.tmp[3] .= q.u[i] .+ α.*dq.u[i]
        r[i] .=  ((q.ds[1] .+ α.*dq.ds[1]).*(dds!(q.u, i, c.tmp[1]) .+ dds!(dq.u, i, c.tmp[2])
                    .- c.F(0.0, c.tmp[3], c.tmp[4])))
    end
    return r
end

compute_residual!(c::Cache,
                  q::PeriodicOrbit,
                 dq::PeriodicOrbit,
                  α::Real=1) = compute_residual!(c, q, dq, c.res, α)
