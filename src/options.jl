# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

using Parameters

export Options

# ~~~ SEARCH OPTIONS FOR NEWTON ITERATIONS ~~~

@with_kw struct Options
    maxiter::Int        = 10     # maximum newton iteration number
    skipiter::Int       = 1      # skip iteration between displays
    verbose::Bool       = true   # print iteration status
    x_tol::Float64      = 1e-10  # tolerance on initial state correction
    r_tol::Float64      = 1e-10  # tolerance on initial state correction
    ls_maxiter::Int     = 10     # maximum number of line search iterations 
    ls_rho::Float64     = 0.5    # line search step reduction factor
end