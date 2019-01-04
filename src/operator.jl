# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

# The operator `J` appearing in the variational approach
struct _Operator{DT, AT, DXT, X, H}
    D::DT   # linearised system operator
    A::AT   # adjoint of the linearised system operator
    Dx::DXT # shift operator
    q::H    # current orbit around which linearisation is taken
    tmp::Tuple{X, X, X}
    _Operator(D, A, Dx, q::H) where {H<:PeriodicOrbit} =
        new{typeof(D), typeof(A), typeof(Dx),
            typeof(q.u[1]), H}(D, A, Dx, q, ntuple(i->similar(q.u[1]), 3))

    _Operator(D, A, q::H) where {H<:PeriodicOrbit} =
        _Operator(D, A, nothing, q::H)
end

# Application of the operator
function Base.A_mul_B!(out::U,
                        op::_Operator{DT, AT, Void},
                         p::PeriodicOrbit{U, 1}) where {DT, AT, M, U<:StateSpaceLoop{M}}
    for i = 1:M
        out[i] .= (  op.q.ds[1] .* dds!(   p.u, i, op.tmp[1])
                   .+   p.ds[1] .* dds!(op.q.u, i, op.tmp[2])
                   .- op.D(0.0, op.q.u[i], p.u[i], op.tmp[3]) )
    end
    return out
end

# And of its adjoint
function Base.At_mul_B!(out::PeriodicOrbit{U, 1},
                         op::_Operator{DT, AT, Void},
                          r::U) where {DT, AT, M, U<:StateSpaceLoop{M}}
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