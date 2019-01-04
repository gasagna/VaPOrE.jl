# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

import HDF5: write, h5open, attrs

export PeriodicOrbit, load!, save

# ~ A PERIODIC TRAJECTORY, PLUS FREQUENCY AND RELATIVE VELOCITY ~
# The parameter U is the space-time function space over which 
# a periodic orbit is defined, while the parameter NS is the
# number of continuous symmetries of the problem, i.e. time plus
# optionally a spatial shift
mutable struct PeriodicOrbit{U<:StateSpaceLoop, NS}
     u::U                   # solution in relative frame
    ds::NTuple{NS, Float64} # natural frequency and optionally relative velocity
    PeriodicOrbit(u::U, ω::Real, v::Real) where {U<:StateSpaceLoop} =
        new{U, 2}(u, (Float64(ω), Float64(v)))
    PeriodicOrbit(u::U, ω::Real) where {U<:StateSpaceLoop} =
        new{U, 1}(u, (Float64(ω),))
end

# ~ OBEY ABSTRACTVECTOR INTERFACE ~
Base.similar(q::PeriodicOrbit) = PeriodicOrbit(similar(q.u), zero.(q.ds)...)
Base.copy(q::PeriodicOrbit) = PeriodicOrbit(copy(q.u), copy.(q.ds)...)

Base.dot(q::PO, p::PO) where {PO<:PeriodicOrbit} =
    dot(q.u, p.u) + sum(q.ds .* p.ds)

Base.norm(q::PeriodicOrbit) = sqrt(dot(q, q))

# ~ BROADCASTING ~
@generated function Base.Broadcast.broadcast!(f, q::PeriodicOrbit, args...)
    quote 
        $(Expr(:meta, :inline))
        broadcast!(f, q.u,  map(_get_u, args)...)
        q.ds = broadcast(f, map(_get_ds, args)...)
        return q
    end
end

_get_u(q::PeriodicOrbit) = q.u; _get_u(q) = q
_get_ds(q::PeriodicOrbit) = q.ds; _get_ds(q) = q


# ~ HDF5 IO ~
function save(q::PeriodicOrbit, path::String)
    # save trajectory to a large matrix first
    data = zeros(Float64, length(q.u[1]), length(q.u))
    for (i, qi) in enumerate(q.u)
        data[:, i] .= qi
    end
    h5open(path, "w") do file
        write(file, "q", data)
        for i = 1:length(q.ds)
            attrs(file)["ds_$i"] = q.ds[i]
        end
    end
end

function load!(x::X, fun, path::String) where {X}
    xs = X[]
    h5open(path, "r") do file
        data = read(file, "q")
        for i = 1:size(data, 2)
            push!(xs, fun(data[:, i]))
        end
        ds = ntuple(i->attrs(file)["ds_$i"], length(attrs(file)))
        return PeriodicOrbit(StateSpaceLoop(xs), ds...)
    end
end