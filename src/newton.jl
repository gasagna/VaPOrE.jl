# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

export nksearch!

# function to calculate the negative redidual
function _residual(x::PeriodicOrbit, F, order::Int)
    tmp1 = similar(x[1])
    tmp2 = similar(x[1])
    function R(out::X, x::PeriodicOrbit{T, X}, i::Int) where {T, X}
        M, N = length(x), length(x[1])
            # sys.D isa Void || (sys.b[_blockrng(i, N)] .-= x.v.*sys.D(sys.tmp, x[i]))
        out .= .- x.ω.*dds!(x, i, tmp1, order) .+ F(0.0, x[i], tmp2)
        return out
    end
end

function nksearch!(q::PeriodicOrbit, order::Int, F, L, D, opts::Options=Options())
    @checkorder order 

    # allocate system
    sys = NRSystem(q, order, F, L, D, _residual(q, F, order))

    # correction
    dq = similar(q)

    # temporaries
    p, dpds, dpdx, f = similar(q), similar(q), similar(q), similar(q)

    # calculate initial error
    r_norm = calc_r_norm(F, D, q, dq, 0, order, p, dpds, dpdx, f)

    # display status if verbose
    opts.verbose && display_header()
    opts.verbose && display_status(0,      # iteration number
                                   0,      # total norm of correction
                                   0,      # period correction
                                   0,      # shift correction
                                   2π/q.ω, # current period
                                   q.v,    # current velocity
                                   r_norm, # error norm after step
                                   0.0)    # step length

    # convergence status
    status = :nr_maxiter_reached

    # newton iterations loop
    for iter = 1:opts.maxiter

        # update Newton-Rapshon system
        update!(sys, q)

        # solve system and write to dq
        solve!(sys, dq)
       
        # perform line search
        λ, r_norm, ls_converged = linesearch(F, D,    q,    dq, order,
                                       p, dpds, dpdx, f,  opts)

        if ls_converged == false
            status = :ls_maxiter_reached
            break 
        end

        # apply correction
        q .+= λ.*dq

        # orbit correction norm
        du_norm = mean(norm, dq)

        # display status if verbose
        if opts.verbose && iter % opts.skipiter == 0 
            display_status(iter,      # iteration number
                           du_norm,   # total norm of correction
                          -dq.ω/2π,   # period correction
                           dq.v,      # velocity correction
                           2π/q.ω,    # new period
                           q.v,       # new velocity
                           r_norm,    # error norm after step
                           λ)         # optimal step length
        end

        # any of the tolerances reached
        if r_norm  < opts.r_tol || du_norm < opts.x_tol
            status = :converged
            break
        end
    end

    # display status if verbose
    if opts.verbose
        display_status(0,         # iteration number
                       0,         # total norm of correction
                       0,         # period correction
                       0,         # velocity correction
                       2π/q.ω,    # new period
                       q.v,       # new shift
                       r_norm,    # error norm after step
                       0)         # optimal step length
    end

    # return input
    return q, status
end