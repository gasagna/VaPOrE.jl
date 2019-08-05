import LinearAlgebra
import FFTW
import Roots

export Plane,
       crossings,
       perinterp,
       project

"""
Plane for Poincare return maps. The point `x` denotes the origin
of the plane, whereas the `n` denotes the normal to the plane. The
latter does not need to be of unit norm, as the sign of dot products
with this vector are what matters.
"""
struct Plane{X}
    n::X
    x::X
    α::Float64 # dot product
    Plane(x::X, n::X) where {X} = new{X}(n, x, LinearAlgebra.dot(x, n))
end

# Compute projection of vector `x` onto the normal of plane `P`, 
# Taking into account the shift of the plane origin. It is used 
# to tell if `x` is below or above the plane to check crossings.
# Equivalent to:
# project(x, P) = (x - x)⋅n = x⋅n - x⋅n = x⋅n - α
project(x::X, P::Plane{X}) where {X} = LinearAlgebra.dot(x, P.n) - P.α

# interpolate periodic data
function perinterp(N::Int, p̂s::AbstractVector{Complex{T}}, s::Real) where {T}
    @inbounds begin
        p = Complex{T}(p̂s[1]/2)
        @simd for k = 2:length(p̂s)-1
            p += p̂s[k]*exp(im*(k-1)*s)
        end
        c = N % 2 == 1 ? 1 : 2
        p += p̂s[end]*exp(im*(length(p̂s)-1)*s)/c
    end
    return 2*real(p)/N
end

# Intersections of a trajectory with the plane, as a vector of 
# floating point indices. E.g., if the first intersection happens 
# exactly in between the first and second points of the trajectory, 
# the first entry of the returned vector will be 1.5. There 
# is a bad corner case that can arise for periodic trajectories, i.e. when 
# the first/last point is exactly on the plane and there will a double 
function crossings(q::PeriodicOrbit{U},
                   P::Plane{X},
                   degree::Int=9) where {M,
                                         ORDER,
                                         T,
                                         X,
                                         U<:StateSpaceLoop{M, ORDER, T, X}}
    # compute projections
    ps = project.(loop(q)._data, Ref(P))

    # define interpolating function for periodic data
    p̂s = FFTW.rfft(ps)
    f(s) = perinterp(M, p̂s, s)

    # find sign changes
    seeds = 2π.*(findall(diff(sign.(ps)) .!= 0) .+ 1)./M

    # maybe add origin 
    # if ps[1]*ps[end] < 0
        # push!(seeds, 0.0)
    # end
    
    # find intersections
    ss = [Roots.find_zero(f, seed, Roots.Order1(), atol=1e-14) for seed in seeds]

    # define storage for interpolation
    store = VaPOrE.makestorage(q, degree)

    # now do interpolations
    xs = [store(similar(loop(q)._data[1]), s/shifts(q)[1], Val(0)) for s in ss]

    return ss, xs
end