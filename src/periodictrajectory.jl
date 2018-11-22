# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of SouthamUon #
# ------------------------------------------------------------------- #

export PeriodicTrajectory, dds!

# Coefficient for finite difference approximation of the loop derivative
const _FDCOEFFS = Dict{Int, Vector{Float64}}()
_FDCOEFFS[2]  = Float64[1]./2
_FDCOEFFS[4]  = Float64[8, -1]./12
_FDCOEFFS[6]  = Float64[45, -9, +1]./60
_FDCOEFFS[8]  = Float64[672, -168, +32, -3]./840
_FDCOEFFS[10] = Float64[2100, -600, +150, -25, +2]./2520

# ~ A PERIODIC TRAJECTORY ~
struct PeriodicTrajectory{M, ORDER, T,
                          X<:AbstractVector{T},
                          V<:AbstractVector{X}} <: AbstractVector{X}
    _data::V # last point omitted
    function PeriodicTrajectory(data::AbstractVector{X}, order::Int) where {X}
        order in (2, 4, 6, 8, 10) ||
            throw(ArgumentError("Error must be in (2, 4, 6, 8, 10)"))
        new{length(data), order, eltype(data[1]), X, typeof(data)}(data)
    end
end

# build from discrete forward map and initial condition
function PeriodicTrajectory(g, M::Int, x::X) where {X}
    xs = X[copy(x)]
    for i = 1:M-1
        push!(xs, g(copy(xs[end])))
    end
    return PeriodicTrajectory(xs)
end

# private
_order(u::PeriodicTrajectory{M, ORDER}) where {M, ORDER} = ORDER

# ~ OBEY ABSTRACTVECTOR INTERFACE ~
@inline Base.@propagate_inbounds function Base.getindex(u::PeriodicTrajectory{M},
                                                i::Integer) where {M}
    @inbounds val = u._data[mod(i-1, M)+1]
    return val
end
@inline Base.@propagate_inbounds function Base.setindex!(u::PeriodicTrajectory{M},
                                                val, i::Integer) where {M}
    @inbounds u._data[mod(i-1, M)+1] = val
    return val
end

@inline Base.length(u::PeriodicTrajectory{M}) where {M} = M
@inline Base.size(u::PeriodicTrajectory{M}) where {M} = (M, )

Base.similar(u::PeriodicTrajectory) = PeriodicTrajectory(similar.(u._data), _order(u))
Base.copy(u::PeriodicTrajectory) = PeriodicTrajectory(copy.(u._data), _order(u))

Base.dot(u::U, v::U) where {U <: PeriodicTrajectory} =
    mapreduce(args->dot(args...), +, zip(u, v))/length(u)

Base.norm(u::PeriodicTrajectory) = sqrt(dot(u, u))

# ~ BROADCASTING ~
@generated function Base.Broadcast.broadcast!(f,
                          u::PeriodicTrajectory{M}, args::Vararg{Any, N}) where {M, N}
    quote
        $(Expr(:meta, :inline))
        for i = 1:$M # loop over elements and apply broadcast
            broadcast!(f, u._data[i],
                $((args[j] <: PeriodicTrajectory ?
                                  :(args[$j][i]) :
                                  :(args[$j]) for j = 1:N)...))
        end
        return u
    end
end

# ~ DERIVATIVE APPROXIMATION ~
# @generated function dds!(u::PeriodicTrajectory{M, ORDER, T, X},
#                                          i::Int, duds::X) where {T, X, M, ORDER}
#     ex = quote duds .= 0 end
#     for (j, c) in enumerate(_FDCOEFFS[ORDER])
#         push!(ex.args, :(duds .+= $c.*u[i+$j] .- $c.*u[i-$j]))
#     end
#     push!(ex.args, :(duds .*= $(M/2π)))
#     push!(ex.args, :(return duds))
#     return ex
# end

function dds!(u::PeriodicTrajectory{M, ORDER, T, X},
                                         i::Int, duds::X) where {T, X, M, ORDER}
    duds .= 0
    for (j, c) in enumerate(_FDCOEFFS[ORDER])
        duds .+= c.*u[i+j] .- c.*u[i-j]
    end
    duds .*= M/2π
    return duds
end