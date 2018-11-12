# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

export GradientCache

struct GradientCache{F, L, X, P<:PeriodicOrbit}
    F::F             # an operator that computes the vector field
    L_adj::L         # an operator that computes the action of the adjoint
    order::Int       # the order for the derivative
    tmp::Tuple{X, X} # temporaries
    r::P             # temporary periodic orbit
end

GradientCache(F, L_adj, x::PeriodicOrbit, order::Int) =
    GradientCache(F, L_adj, order, (similar(x[1]), similar(x[1])), similar(x))

# compute gradient or mean square residual
function (cache::GradientCache)(grad::PeriodicOrbit, x::PeriodicOrbit)
    # reset gradient, so we can update it
    grad.ω = 0
    grad.v = 0

    # we'll use this later
    M = length(x)

    # compute residual first, along with gradient wrt to ω
    for i = 1:M
        # calc tangent vector
        dds!(x, i, cache.tmp[1], cache.order)
        
        # calc residual
        cache.r[i] .= x.ω.*cache.tmp[1] .- cache.F(0.0, x[i], cache.tmp[2])
        
        # calc gradient 
        grad.ω += dot(cache.tmp[1], cache.r[i])/M
    end

    # now calc gradient
    for i = 1:M
        grad[i] .= (.- x.ω .* dds!(cache.r, i, cache.tmp[1], cache.order)
                    .- cache.L_adj(0.0, x[i], cache.r[i], cache.tmp[2]))
    end

    return grad
end
