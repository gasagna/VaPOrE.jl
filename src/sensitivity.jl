# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

export sensitivity!

function sensitivity!(y::PeriodicOrbit, x::PeriodicOrbit, F, L, D, R, order)
    sys = NRSystem(x, order, F, L, D, R)
    return solve!(update!(sys, x), y)
end