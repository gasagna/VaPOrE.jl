# ------------------------------------------------------------------- #
# Copyright 2017-2019, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
import LinearAlgebra: dot, norm

import HDF5: write, h5open, h5readattr, attrs

export PeriodicOrbit, load!, save, toorder, loop, shifts

# ~ A PERIODIC TRAJECTORY, PLUS FREQUENCY AND RELATIVE VELOCITY ~
# The parameter U is the space-time function space over which 
# a periodic orbit is defined, while the parameter NS is the
# number of continuous symmetries of the problem, i.e. time plus
# optionally a spatial shift
mutable struct PeriodicOrbit{U<:StateSpaceLoop, NS} <: AbstractVector{Float64}
     u::U                   # solution in relative frame
    ds::NTuple{NS, Float64} # natural frequency and optionally relative velocity
    PeriodicOrbit(u::U, ω::Real, v::Real) where {U<:StateSpaceLoop} =
        new{U, 2}(u, (Float64(ω), Float64(v)))
    PeriodicOrbit(u::U, ω::Real) where {U<:StateSpaceLoop} =
        new{U, 1}(u, (Float64(ω),))
end

# accessors functions
loop(q::PeriodicOrbit) = q.u
shifts(q::PeriodicOrbit) = q.ds

# ~ OBEY ABSTRACTVECTOR INTERFACE ~
Base.similar(q::PeriodicOrbit) = PeriodicOrbit(similar(q.u), zero.(q.ds)...)
Base.copy(q::PeriodicOrbit) = PeriodicOrbit(copy(q.u), copy.(q.ds)...)

Base.getindex(q::PeriodicOrbit, i::Int) = q.u[i]

# these are hacks!
@inline Base.length(q::PeriodicOrbit) = length(q.u) + length(q.ds)
@inline Base.size(q::PeriodicOrbit) = (length(q), )

dot(q::PO, p::PO) where {PO<:PeriodicOrbit} = dot(q.u, p.u) + sum(q.ds .* p.ds)

norm(q::PeriodicOrbit) = sqrt(dot(q, q))

# change order
toorder(q::PeriodicOrbit, order::Int) = 
    PeriodicOrbit(StateSpaceLoop(q.u._data, order), q.ds...)

# ~ BROADCASTING ~
const POStyle = Broadcast.ArrayStyle{PeriodicOrbit}
Base.BroadcastStyle(::Type{<:PeriodicOrbit}) = Broadcast.ArrayStyle{PeriodicOrbit}()

@inline function Base.copyto!(dest::PeriodicOrbit,
                                bc::Broadcast.Broadcasted{POStyle})
    copyto!(loop(dest), po_unpack(bc, Val(:loop)))
    dest.ds = broadcast(bc.f, po_unpack(bc, Val(:shifts)))
    return dest
end

@inline function Base.copyto!(dest::PeriodicOrbit, src::PeriodicOrbit)
    copyto!(loop(dest), loop(src))
    dest.ds = src.ds
    return dest
end

@inline po_unpack(bc::Broadcast.Broadcasted, item::Val) =
    Broadcast.Broadcasted(bc.f, _po_unpack(bc.args, item))

@inline po_unpack(x::PeriodicOrbit, ::Val{:loop})   = loop(x)
@inline po_unpack(x::PeriodicOrbit, ::Val{:shifts}) = shifts(x)
@inline po_unpack(x::Any, ::Val) = x

@inline _po_unpack(args::Tuple, item) = 
    (po_unpack(args[1], item), _po_unpack(Base.tail(args), item)...)
@inline _po_unpack(args::Tuple{Any}, item) = (po_unpack(args[1], item),)
@inline _po_unpack(args::Tuple{}, item) = ()

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
        attrs(file)["order"] = order(loop(q))
    end
end

function load!(x::X, fun, path::String; defaultorder::Int=10) where {X}
    xs = X[]
    h5open(path, "r") do file
        data = read(file, "q")
        for i = 1:size(data, 2)
            push!(xs, fun(data[:, i]))
        end
    end
    atrs = h5readattr(path, "/")
    order = "order" in keys(atrs) ? atrs["order"] : defaultorder
    nshifts = "ds_2" in keys(atrs) ? 2 : 1
    return PeriodicOrbit(StateSpaceLoop(xs, order), [atrs["ds_$i"] for i in 1:nshifts]...)
end