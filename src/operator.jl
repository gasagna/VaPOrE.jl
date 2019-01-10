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
    tmp::NTuple{5, X} # temporaries
    _Operator(L, A, D, q::H) where {H<:PeriodicOrbit} =
        new{typeof(L), typeof(A), typeof(D),
            typeof(q.u[1]), H}(L, A, D, q, ntuple(i->similar(q.u[1]), 5))

    _Operator(L, A, q::H) where {H<:PeriodicOrbit} =
        _Operator(L, A, nothing, q::H)
end

# Application of the operator
function LinearAlgebra.mul!(out::U,
                             op::_Operator,
                              p::PeriodicOrbit{U}) where {M, U<:StateSpaceLoop{M}}
   if op.D isa Nothing
        for i = 1:M
            out[i] .= (  op.q.ds[1] .* dds!(   p.u, i, op.tmp[1])
                       .+   p.ds[1] .* dds!(op.q.u, i, op.tmp[2])
                       .- op.L(0.0, op.q.u[i], p.u[i], op.tmp[3]) )
        end
    else
        for i = 1:M
            out[i] .= (  op.q.ds[1] .* dds!(   p.u, i, op.tmp[1])
                       .+   p.ds[1] .* dds!(op.q.u, i, op.tmp[2])
                       .+op.q.ds[2] .* op.D(op.tmp[3],    p.u[i])
                       .+   p.ds[2] .* op.D(op.tmp[4], op.q.u[i])
                       .- op.L(0.0, op.q.u[i], p.u[i], op.tmp[5]) )
        end
    end
    return out
end

# And of its adjoint
function LinearAlgebra.mul!(out::PeriodicOrbit{U},
                             op::_Operator,
                              r::U, ::ADJOINT) where {M, U<:StateSpaceLoop{M}}
    if op.D isa Nothing
        # initialise to a zero of appropriate arithmetic type
        ds1 = zero(dot(dds!(op.q.u, 1, op.tmp[2]), r[1])/M)
        for i = 1:M
            # calculate the dot product manuallly
            ds1 += dot(dds!(op.q.u, i, op.tmp[2]), r[i])/M
            out.u[i] .= (.- op.q.ds[1] .* dds!(r, i, op.tmp[1])
                         .- op.A(0.0, op.q.u[i], r[i], op.tmp[2]) )
        end
        out.ds = (ds1, )
    else
        # initialise to a zero of appropriate arithmetic type
        ds1 = zero(dot(dds!(op.q.u, 1, op.tmp[1]), r[1])/M)
        ds2 = zero(dot(op.D(op.tmp[1], op.q.u[1]), r[1])/M)
        for i = 1:M
            # calculate the dot product manuallly
            ds1 += dot(dds!(op.q.u, i, op.tmp[1]), r[i])/M
            ds2 += dot(op.D(op.tmp[1], op.q.u[i]), r[i])/M
            out.u[i] .= (.- op.q.ds[1] .* dds!(r, i, op.tmp[1])
                         .- op.q.ds[2] .* op.D(op.tmp[2], r[i])
                         .- op.A(0.0, op.q.u[i], r[i], op.tmp[3]) )
        end
        out.ds = (ds1, ds2)
    end
    return out
end