# ------------------------------------------------------------------- #
# Copyright 2017-2019, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

export tangent

# version with no shifts
function tangent(x::PeriodicOrbit{U, 1}, 
                 F, # right hand side of nonlinear equations
                 L, # linearisation of F
                 R) where {U} # forcing for tangent sensitivity R(out, x)
    # construct a cache object on the current solution
    cache = Cache(x, F, L, nothing, nothing)
    update!(x, cache, false)

    # get problem size
    M, N = length(x.u), length(x.u[1])
    
    # now construct right hand side
    tmp = similar(loop(x)[1])
    for i in 1:M
        rng = _blockrng(i, N)
        cache.b[rng] .= R(tmp, loop(x)[i])
    end
    cache.b[end] = 0

    # solve linear system
    _y = cache.A \ cache.b;

    # pack into periodic orbit
    y = similar(x)
    for i in 1:M
        y[i] .= _y[_blockrng(i, N)]
    end

    # note the right column is du/ds
    return y, -_y[end]/shifts(x)[1]
end