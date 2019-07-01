# ------------------------------------------------------------------- #
# Copyright 2017-2019, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
import Flows: samples, AbstractStorage, RAMStorage

export jacobians

struct JacobianOp{T, OP, CACHE} <: AbstractMatrix{T}
       op::OP
    cache::CACHE
     span::Tuple{Real, Real}
    function JacobianOp(op::OP,
                        cache::CACHE,
                        span::Tuple{Real, Real}) where {OP, CACHE<:AbstractStorage}
        new{eltype(samples(cache)[1]), OP, CACHE}(op, cache, span)
    end
end

function Base.size(jac::JacobianOp)
    n = length(samples(jac.cache)[1])
    return (n, n)
end

function LinearAlgebra.mul!(out::AbstractVector,
                            jac::JacobianOp,
                            x::AbstractVector)
    out .= x
    return jac.op(out, jac.cache, jac.span)
end

"""
    ith_span(i::Int, N::Int, T::Real)

Split the interval `(0, T)` in `N` equal sub-intervals and 
return the `i`-th one as a 2-tuple. Make sure that the last
span does not go out of bounds `(0, T)``
"""
ith_span(i::Int, N::Int, T::Real) =  ((i-1)*T/N, min(i*T/N, T))

"""
    jacobians(ψ, q::PeriodicOrbit, N::Int)

Construct a `N`-tuple of `JacobianOp` objects that calculate the action
of the jacobian matrix on vectors in tangent space, using the linearised
flow operator `ψ`. The `i`-th element calculates the action over
the time span `(i-1)*T/N to `i*T/N`, where `T` is the orbit period.
With `N=1`, one obtains an object that calculates the action of the
monodromy matrix in a matrix-free fashion.
"""
function jacobians(ψ, q::PeriodicOrbit, N::Int, degree::Int, getcache::Bool=false)
    cache = makecache(q, N, degree)
    Js = ntuple(i->JacobianOp(ψ,
                                cache,
                                ith_span(i, N, 2π/shifts(q)[1])), N)
    return getcache == true ? (Js, cache) : Js
end


function makecache(q::PeriodicOrbit, N::Int, degree::Int)
    # period andlength of loop
    T = 2π/shifts(q)[1]
    M = length(loop(q))

    # init cache
    cache = RAMStorage(eltype(loop(q)); degree=degree, period=T)

    # push elements to cache
    for i = 1:M
        push!(cache, (i-1)*T/M, loop(q)[i])
    end

    return cache
end