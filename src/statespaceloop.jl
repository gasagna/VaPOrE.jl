# ------------------------------------------------------------------- #
# Copyright 2017-2019, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
import LinearAlgebra: dot, norm

export StateSpaceLoop, dds!, prolong, restrict, order

# Coefficient for finite difference approximation of the loop derivative
const _FDCOEFFS = Dict{Int, Vector{Float64}}()
_FDCOEFFS[2]  = Float64[1]./2
_FDCOEFFS[4]  = Float64[8, -1]./12
_FDCOEFFS[6]  = Float64[45, -9, +1]./60
_FDCOEFFS[8]  = Float64[672, -168, +32, -3]./840
_FDCOEFFS[10] = Float64[2100, -600, +150, -25, +2]./2520

# ~ A PERIODIC TRAJECTORY ~
struct StateSpaceLoop{M, ORDER, T,
                          X<:AbstractVector{T},
                          V<:AbstractVector{X}} <: AbstractVector{X}
    _data::V # last point omitted
    function StateSpaceLoop(data::AbstractVector{X}, order::Int) where {X}
        order in (2, 4, 6, 8, 10) ||
            throw(ArgumentError("Error must be in (2, 4, 6, 8, 10)"))
        new{length(data), order, eltype(data[1]), X, typeof(data)}(data)
    end
end

# build from discrete forward map and initial condition
function StateSpaceLoop(g, M::Int, x::X, order::Int) where {X}
    xs = X[copy(x)]
    for i = 1:M-1
        push!(xs, g(copy(xs[end])))
    end
    return StateSpaceLoop(xs, order)
end

# private
order(u::StateSpaceLoop{M, ORDER}) where {M, ORDER} = ORDER

# Use linear interpolation to prolong a loop TODO: use a better interpolation
function prolong(x::StateSpaceLoop{M, ORDER}) where {M, ORDER}
    out = StateSpaceLoop([similar(x[1]) for i = 1:2*M], ORDER)
    for i = 1:M
        out[2*i-1] .= x[i]
        out[2*i]   .= 0.5.*(x[i] .+ x[i+1])
    end
    return out
end

function restrict(x::StateSpaceLoop{M, ORDER}) where {M, ORDER}
    M % 2 == 0 || throw(ArgumentError("loop length must be even"))
    return StateSpaceLoop(x._data[1:2:end], ORDER)
end

# ~ OBEY ABSTRACTVECTOR INTERFACE ~
@inline Base.@propagate_inbounds function Base.getindex(u::StateSpaceLoop{M},
                                                i::Integer) where {M}
    @inbounds val = u._data[mod(i-1, M)+1]
    return val
end
@inline Base.@propagate_inbounds function Base.setindex!(u::StateSpaceLoop{M},
                                                val, i::Integer) where {M}
    @inbounds u._data[mod(i-1, M)+1] = val
    return val
end

@inline Base.length(u::StateSpaceLoop{M}) where {M} = M
@inline Base.size(u::StateSpaceLoop{M}) where {M} = (M, )

Base.similar(u::StateSpaceLoop) = StateSpaceLoop(similar.(u._data), order(u))
Base.copy(u::StateSpaceLoop) = StateSpaceLoop(copy.(u._data), order(u))

dot(u::U, v::U) where {U <: StateSpaceLoop} =
    mapreduce(args->dot(args...), +, zip(u, v))/length(u)

norm(u::StateSpaceLoop) = sqrt(dot(u, u))

# ~ BROADCASTING ~
# taken from MultiscaleArrays.jl
const SSLStyle = Broadcast.ArrayStyle{StateSpaceLoop}
Base.BroadcastStyle(::Type{<:StateSpaceLoop}) = Broadcast.ArrayStyle{StateSpaceLoop}()

@inline function Base.copyto!(dest::StateSpaceLoop{M},
                                bc::Broadcast.Broadcasted{SSLStyle}) where {M}
    for i in 1:M
        copyto!(dest._data[i], ssl_unpack(bc, i))
    end
    return dest
end

@inline ssl_unpack(bc::Broadcast.Broadcasted, i) =
    Broadcast.Broadcasted(bc.f, _ssl_unpack(i, bc.args))
@inline ssl_unpack(x, ::Any) = x
@inline ssl_unpack(x::StateSpaceLoop, i) = x._data[i]

@inline _ssl_unpack(i, args::Tuple) = 
    (ssl_unpack(args[1], i), _ssl_unpack(i, Base.tail(args))...)
@inline _ssl_unpack(i, args::Tuple{Any}) = (ssl_unpack(args[1], i),)
@inline _ssl_unpack(::Any, args::Tuple{}) = ()

# ~ DERIVATIVE APPROXIMATION ~
@generated function dds!(u::StateSpaceLoop{M, ORDER, T, X},
                         i::Int, 
                      duds::X) where {T, X, M, ORDER}
    ex = quote duds .= 0 end
    for (j, c) in enumerate(_FDCOEFFS[ORDER])
        push!(ex.args, :(duds .+= $c.*u[i+$j] .- $c.*u[i-$j]))
    end
    push!(ex.args, :(duds .*= $(M/2π)))
    push!(ex.args, :(return duds))
    return ex
end