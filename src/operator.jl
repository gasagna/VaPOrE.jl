# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

struct _Operator{LT, L⁺T, X, H}
    L::LT
    L⁺::L⁺T
    q::H
    tmp::Tuple{X, X, X}
    _Operator(L, L⁺, q::H) where {H<:PeriodicOrbit} =
        new{typeof(L), typeof(L⁺),
            typeof(q.u[1]), H}(L, L⁺, q, ntuple(i->similar(q.u[1]), 3))
end

function Base.A_mul_B!(out::U,
                        op::_Operator,
                         p::PeriodicOrbit{U}) where {U<:PeriodicTrajectory}
    for i = 1:length(out)
        out[i] .= (  op.q.ω .* dds!(   p.u, i, op.tmp[1])
                   .+   p.ω .* dds!(op.q.u, i, op.tmp[2])
                   .- op.L(0.0, op.q.u[i],    p.u[i], op.tmp[3]) )
    end
    return out
end

function Base.At_mul_B!(out::PeriodicOrbit{U},
                         op::_Operator,
                         r::U) where {U<:PeriodicTrajectory}
    M = length(r)
    out.ω = 0
    out.v = 0
    for i = 1:M
        # calculate the dot product manuallly
        out.ω += dot(dds!(op.q.u, i, op.tmp[2]), r[i])/M
        out.u[i] .= (.- op.q.ω .* dds!(r, i, op.tmp[1])
                     .- op.L⁺(0.0, op.q.u[i], r[i], op.tmp[2]) )
    end
    return out
end