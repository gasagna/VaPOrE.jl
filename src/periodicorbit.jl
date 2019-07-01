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

# these are hacks!
@inline Base.length(q::PeriodicOrbit) = length(q.u) + length(q.ds)
@inline Base.size(q::PeriodicOrbit) = (length(q), )

dot(q::PO, p::PO) where {PO<:PeriodicOrbit} = dot(q.u, p.u) + sum(q.ds .* p.ds)

norm(q::PeriodicOrbit) = sqrt(dot(q, q))

# change order
toorder(q::PeriodicOrbit, order::Int) = 
    PeriodicOrbit(StateSpaceLoop(q.u._data, order), q.ds...)

# ~ BROADCASTING ~
# @generated function Base.Broadcast.broadcast!(f, q::PeriodicOrbit, args...)
#     quote 
#         $(Expr(:meta, :inline))
#         broadcast!(f, q.u,  map(_get_u, args)...)
#         q.ds = broadcast(f, map(_get_ds, args)...)
#         return q
#     end
# end

# _get_u(q::PeriodicOrbit) = q.u; _get_u(q) = q
# _get_ds(q::PeriodicOrbit) = q.ds; _get_ds(q) = q

const POStyle = Broadcast.ArrayStyle{PeriodicOrbit}
Base.BroadcastStyle(::Type{<:PeriodicOrbit}) = Broadcast.ArrayStyle{PeriodicOrbit}()
Base.BroadcastStyle(::Broadcast.ArrayStyle{PeriodicOrbit},
                    ::Broadcast.DefaultArrayStyle{1}) = Broadcast.DefaultArrayStyle{1}()
Base.BroadcastStyle(::Broadcast.DefaultArrayStyle{1},
                    ::Broadcast.ArrayStyle{PeriodicOrbit}) = Broadcast.DefaultArrayStyle{1}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PeriodicOrbit}},
               ::Type{E}) where E = similar(bc)

function Base.copyto!(dest::PeriodicOrbit, bc::Broadcast.Broadcasted{POStyle})
    ret = Broadcast.flatten(bc)
    return __broadcast!(ret.f, dest, ret.args...)
end

function __broadcast!(f, A::PeriodicOrbit, Bs::Union{Number, PeriodicOrbit}...)
    broadcast!(f, A.u,  (typeof(B)<:PeriodicOrbit ? B.u  : B for B in Bs)...)
    A.ds = broadcast(f, (typeof(B)<:PeriodicOrbit ? B.ds : B for B in Bs)...)
    return A
end

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
    end
    atrs = h5readattr(path, "/")
    return PeriodicOrbit(StateSpaceLoop(xs, 10), [atrs["ds_$i"] for i in 1:length(atrs)]...)
end