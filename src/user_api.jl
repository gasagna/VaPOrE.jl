# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

export nksearch!

function nksearch!(q::PeriodicOrbit, order::Int, F, L, opts::Options=Options())
    # check differentiation order
    @checkorder order

    # allocate cache (does not update it)
    c = Cache(q, order, F, L, L)

    # define correction
    dq = similar(q)

    # initialise trust region radius and iteration couter
    tr_radius = opts.init_tr_radius
    iter      = 0
    status    = :none

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ITERATIONS LOOP
    while true
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # update the cache with the current solution
        update!(cache, q)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SOLVE TRUST REGION PROBLEM
        hits_boundary = solve_tr_subproblem!(dq, cache, tr_radius)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # cALCULATE RATIO OF ACTUAL AND PREDICTED REDUCTIONS
        # calc predicted value 
        predicted_reduction = 1

        # calc actual reduction
        actual_reduction = 2

        ρ = actual_reduction/predicted_reduction

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TRUST REGION UPDATE
        if ρ < 1/4
            tr_radius *= 1/4
        elseif ρ > 3/4 && hits_boundary
            tr_radius = min(2*tr_radius, opts.max_trust_radius)
        end

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SOLUTION UPDATE
        # apply step only if ratio of actual and predicted reductions 
        # is larger than a user defined η
        if ρ > opts.η
            q .= q .+ dq
        end 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # check tolerances

        # solution update is small 
        if norm(dq) < opts.dq_norm_tol
            status = :converged_dq_norm_tol
            break
        end

        # residual norm is small
        if norm(cache.r)< opts.r_norm_tol
            status = :converged_r_norm_tol
            break
        end

        # ran too many iterations
        if iter >= opts.maxiter
            status = :maxiter_reached
            break
        end

        # update counter
        iter += 1
    end

    return q, status
end