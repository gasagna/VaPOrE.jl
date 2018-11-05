# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

export NKMatrix

# ~~~ MATRIX OPERATOR FOR NEWTON KRYLOV APPROACH ~~~
mutable struct NKMatrix{X, N, FT, LT, GXT, PO<:PeriodicOrbit{X, N}}
      F::FT  # time derivative of forward map Gt(u, ut)
      L::LT  # linear operator
     Gx::GXT # shift derivative of forward map Gx(u, ux)
      q::PO  # current solution
    tmp::X   # temporary
    NKMatrix(F, L, Gx, q::PeriodicOrbit{X, N}, tmp::X) where {X, N} =
        new{X, N, typeof(F), typeof(L), typeof(Gx), typeof(q)}(F, L, Gx, q, tmp)
end

# obey GMRES interface
Base.:*(M::NKMatrix{X, N}, p::PeriodicOrbit{X, N}) where {X, N} = 
    A_mul_B!(similar(p), M, p)

function Base.A_mul_B!(o::PeriodicOrbit{X, N}, M::NKMatrix{X, N}, p::PeriodicOrbit{X, N}) where {X, N}
    for i in 1:N
        # temporal differentiation
        o[i] .= M.q.ω.*dds!(p, i, o[i])

        # subtract linear operator
        o[i] .-= M.L(0.0, M.q[i], p[i], M.tmp)
        # o[i] .+= M.q.v .* M.Gx(p[i], M.tmp)

        # add space derivative of current solution
        # o[i] .+= p.v .* M.Gx(M.q[i], M.tmp)

        # add time derivative of current solution
        o[i] .+= p.ω .* dds!(M.q, i, M.tmp)
    end

    # compute other bits
    # o.v = dot(p[1], M.Gx(    M.q[1], M.tmp))
    o.ω = dot(p[1], M.F(0.0, M.q[1], M.tmp))

    return o
end