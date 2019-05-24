# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

using Parameters

export Options

# ~~~ SEARCH OPTIONS FOR NEWTON ITERATIONS ~~~

@with_kw struct Options
    maxiter::Int              = 10     # maximum newton iteration number
    skipiter::Int             = 1      # skip iteration between displays
    verbose::Bool             = true   # print iteration status
    dq_norm_tol::Float64      = 1e-7   # tolerance on initial state correction
    r_norm_tol::Float64       = 1e-7   # tolerance on initial state correction
    min_step::Float64         = 1e-4   # 
    init_Δ::Float64           = 1      # initial trust region radius
    max_Δ::Float64            = 10^8   #
    η::Float64                = 0
end