# ------------------------------------------------------------------- #
# Copyright 2017-2019, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
import LinearAlgebra: dot, norm

export StateSpaceLoop, dds!, prolong

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
# @generated function Base.Broadcast.broadcast!(f,
#                           u::StateSpaceLoop{M}, args::Vararg{Any, N}) where {M, N}
#     quote
#         $(Expr(:meta, :inline))
#         for i = 1:$M # loop over elements and apply broadcast
#             broadcast!(f, u._data[i],
#                 $((args[j] <: StateSpaceLoop ?
#                                   :(args[$j][i]) :
#                                   :(args[$j]) for j = 1:N)...))
#         end
#         return u
#     end
# end


const SSLStyle = Broadcast.ArrayStyle{StateSpaceLoop}
Base.BroadcastStyle(::Type{<:StateSpaceLoop}) = Broadcast.ArrayStyle{StateSpaceLoop}()
Base.BroadcastStyle(::Broadcast.ArrayStyle{StateSpaceLoop},
                    ::Broadcast.DefaultArrayStyle{1}) = Broadcast.DefaultArrayStyle{1}()
Base.BroadcastStyle(::Broadcast.DefaultArrayStyle{1},
                    ::Broadcast.ArrayStyle{StateSpaceLoop}) = Broadcast.DefaultArrayStyle{1}()

function Base.copyto!(dest::StateSpaceLoop, bc::Broadcast.Broadcasted{SSLStyle})
    ret = Broadcast.flatten(bc)
    return __broadcast!(ret.f, dest, ret.args...)
end

function __broadcast!(f, A::StateSpaceLoop{M}, Bs::Union{Number, StateSpaceLoop}...) where {M}
    for i in 1:M
        broadcast!(f, A._data[i], (typeof(B)<:StateSpaceLoop ? B._data[i] : B for B in Bs)...)
    end
    return A
end

# ~ DERIVATIVE APPROXIMATION ~
# @generated function dds!(u::StateSpaceLoop{M, ORDER, T, X},
#                                          i::Int, duds::X) where {T, X, M, ORDER}
#     ex = quote duds .= 0 end
#     for (j, c) in enumerate(_FDCOEFFS[ORDER])
#         push!(ex.args, :(duds .+= $c.*u[i+$j] .- $c.*u[i-$j]))
#     end
#     push!(ex.args, :(duds .*= $(M/2π)))
#     push!(ex.args, :(return duds))
#     return ex
# end

function dds!(u::StateSpaceLoop{M, ORDER, T, X},
                                         i::Int, duds::X) where {T, X, M, ORDER}
    duds .= 0
    for (j, c) in enumerate(_FDCOEFFS[ORDER])
        duds .+= c.*u[i+j] .- c.*u[i-j]
    end
    duds .*= M/2π
    return duds
end