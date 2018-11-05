# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

import HDF5: write, h5open, attrs

export PeriodicOrbit, dds!, load!, save

# ~ A PERIODIC TRAJECTORY ~
mutable struct PeriodicOrbit{T, X<:AbstractVector{T}} <: AbstractVector{X}
    u::Vector{X} # solution in relative frame, last point omitted
    ω::Float64   # natural frequency
    v::Float64   # relative velocity
end

# ~ outer constructors ~
PeriodicOrbit(data::Vector{<:AbstractVector}, ω::Real, v::Real) =
    PeriodicOrbit{eltype(data[1]),
                  eltype(data)}(data, ω, v)

# build from discrete forward map and initial condition
function PeriodicOrbit(g, N::Int, x::X, ω::Real, v::Real) where {X}
    xs = X[copy(x)]
    for i = 1:N-1
        push!(xs, g(copy(xs[end])))
    end
    return PeriodicOrbit(xs, ω, v)
end

# ~ obey AbstractVector interface ~
@inline Base.getindex(q::PeriodicOrbit, i::Integer) = 
    q.u[mod(i-1, length(q))+1]
@inline Base.setindex!(q::PeriodicOrbit, val, i::Integer) =
                      (q.u[mod(i-1, length(q))+1] = val; return val)

Base.size(q::PeriodicOrbit) = (length(q.u), )
Base.similar(q::PeriodicOrbit) = 
    PeriodicOrbit([similar(q.u[1]) for i = 1:length(q)], 0.0, 0.0)
Base.copy(q::PeriodicOrbit) = PeriodicOrbit(copy.(q.u), q.ω, q.v)

# ~ broadcasting ~
@generated function Base.Broadcast.broadcast!(f, q::PeriodicOrbit, args::Vararg{Any, n}) where {n}
    quote 
        $(Expr(:meta, :inline))
        broadcast!(f, q.u, map(_get_u, args)...)
        q.ω = broadcast(f, map(_get_ω, args)...)
        q.v = broadcast(f, map(_get_v, args)...)
        return q
    end
end

_get_u(q::PeriodicOrbit) = q.u; _get_u(q) = q
_get_ω(q::PeriodicOrbit) = q.ω; _get_ω(q) = q
_get_v(q::PeriodicOrbit) = q.v; _get_v(q) = q

# ~ derivative approximation ~
function dds!(q::PeriodicOrbit{T, X}, i::Int, dqdt::X, order::Int) where {T, X}
    @checkorder order
    dqdt .= 0
    for (j, c) in enumerate(_coeffs[order])
        dqdt .+= c.*q[i+j] .- c.*q[i-j]
    end
    dqdt ./= (2π./length(q))
    return dqdt
end

# ~ save orbit to file ~
function save(x::PeriodicOrbit, path::String)
    # save to a large matrix
    data = zeros(Float64, length(x[1]), length(x))
    for (i, xi) in enumerate(x)
        data[:, i] .= xi
    end
    h5open(path, "w") do file
        write(file, "x", data)
        attrs(file)["omega"] = x.ω
        attrs(file)["v"] = x.v
    end
end

# ~ load orbit from file ~
function load!(::X, fun, path::String) where {X}
    xs = X[]
    h5open(path, "r") do file
        data = read(file, "x")
        for i = 1:size(data, 2)
            push!(xs, fun(data[:, i]))
        end
        return PeriodicOrbit(xs, read(attrs(file)["omega"]),
                                 read(attrs(file)["v"]))
    end
end