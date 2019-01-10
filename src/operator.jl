# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
import LinearAlgebra

# The operator `J` appearing in the variational approach
struct _Operator{LT, AT, DXT, X, H}
    L::LT   # linearised system operator
    A::AT   # adjoint of the linearised system operator
    D::DXT  # derivative operator
    q::H    # current orbit around which linearisation is taken
    tmp::Tuple{X, X, X}
    _Operator(L, A, D, q::H) where {H<:PeriodicOrbit} =
        new{typeof(L), typeof(A), typeof(D),
            typeof(q.u[1]), H}(L, A, D, q, ntuple(i->similar(q.u[1]), 3))

    _Operator(L, A, q::H) where {H<:PeriodicOrbit} =
        _Operator(L, A, nothing, q::H)
end

# Application of the operator
function LinearAlgebra.mul!(out::U,
                             op::_Operator{LT, AT, Nothing},
                              p::PeriodicOrbit{U, 1}) where {LT, AT, M, U<:StateSpaceLoop{M}}
    for i = 1:M
        out[i] .= (  op.q.ds[1] .* dds!(   p.u, i, op.tmp[1])
                   .+   p.ds[1] .* dds!(op.q.u, i, op.tmp[2])
                   .- op.L(0.0, op.q.u[i], p.u[i], op.tmp[3]) )
    end
    return out
end

# And of its adjoint
function LinearAlgebra.mul!(out::PeriodicOrbit{U, 1},
                             op::_Operator{LT, AT, Nothing},
                              r::U, ::ADJOINT) where {LT, AT, M, U<:StateSpaceLoop{M}}
    # initialise to a zero of appropriate arithmetic type
    ds1 = zero(dot(dds!(op.q.u, 1, op.tmp[2]), r[1])/M)
    for i = 1:M
        # calculate the dot product manuallly
        ds1 += dot(dds!(op.q.u, i, op.tmp[2]), r[i])/M
        out.u[i] .= (.- op.q.ds[1] .* dds!(r, i, op.tmp[1])
                     .- op.A(0.0, op.q.u[i], r[i], op.tmp[2]) )
    end
    out.ds = (ds1, )
    return out
end