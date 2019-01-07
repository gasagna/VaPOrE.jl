# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

export search!

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# User API
# if we have no spatial shift, we can avoid passing Dx
search!(q::PeriodicOrbit{U, 1}, F, L, L⁺, opts::Options=Options()) where {U} =
    _search!(q, F, L, L⁺, nothing, opts)

# otherwise we have to!
search!(q::PeriodicOrbit{U, 2}, F, L, L⁺, Dx, opts::Options=Options()) where {U} =
    _search!(q, F, L, L⁺, Dx, opts)

# catchall
search!(args...) = 
    throw(ArgumentError("invalid input arguments"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Actual implementation
function _search!(q, F, L, L⁺, Dx, opts)
    # allocate cache (does not update it)
    cache = Cache(q, F, L, L⁺, Dx)

    # define correction
    dq = similar(q)

    # initialise trust region radius and iteration counter
    Δ = opts.init_Δ
    iter      = 0
    status    = :none

    opts.verbose && display_header_tr()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ITERATIONS LOOP
    while true

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SOLVE TRUST REGION PROBLEM
        hits_boundary, which, step = solve_tr_problem!(q, dq, cache, Δ)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # CALCULATE RATIO OF ACTUAL AND PREDICTED REDUCTIONS 
        # calc original residual 
        compute_residual!(cache, q, cache.res)
        a = 0.5*norm(cache.res)^2
        
        # calc residual of perturbed orbit
        compute_residual!(cache, q, dq, cache.res_tmp)
        b = 0.5*norm(cache.res_tmp)^2

        # calc actual value
        actual_reduction = a - b

        # calc predicted reduction
        mul!(cache.den, cache.op, dq)
        c = dot(cache.res, cache.den) 
        d = 0.5*norm(cache.den)^2
        predicted_reduction = - c - d

        # calc ratio
        ρ = actual_reduction/predicted_reduction

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TRUST REGION UPDATE. 
        # This is based on the classical algorithm revised to account 
        # for catastrophic cancellation when the terms in the 
        # "predicted_reduction' and "actual_reduction" are small 
        # quantities and the differences are numerically inaccurate. 
        # This is simply based on the idea that when the residual
        # residual is smaller than a multiple of sqrt of machine accuracy
        # we simply assume that the newton direction will be a good pick.
        # See "A note on robust descent in differentiable optimization"
        # from "Jean-Pierre Dussault"
        if a > 1e-7
            if ρ < 1/4
                Δ *= 1/4
            elseif ρ > 3/4 && hits_boundary
                Δ = min(3*Δ, opts.max_Δ)
            end

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # SOLUTION UPDATE
            # apply step only if ratio of actual and predicted reductions 
            # is larger than a user defined η
            if ρ > opts.η
                q .= q .+ dq
            end 
        else 
            q .= q .+ dq
        end

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Print output
        if opts.verbose
            compute_residual!(cache, q, cache.res)
            display_status_tr(iter, which, step, norm(cache.res), ρ, Δ, norm(dq), dq.ds[1], q.ds[1])
        end

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # CHECK TOLERANCES

        # solution update is small 
        if norm(dq) < opts.dq_norm_tol
            status = :converged_dq_norm_tol
            break
        end

        # residual norm is small
        if norm(cache.res)< opts.r_norm_tol
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