# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

function solve_tr_problem!(q::PeriodicOrbit, dq::PeriodicOrbit, cache::Cache, Δ::Real)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update cache with common computations
    update!(q, cache)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Newton step first
    solve_newton!(cache)
    
    # If the Newton step is within the trust region return
    # this point, which is hopefully a good descent step
    if norm(cache.dq_newton) < Δ
        dq .= cache.dq_newton
        return false, :newton, 1.0
    end

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # If not compute cauchy point, the minimizer along
    # the steepest descent direction
    solve_cauchy!(q, cache)

    # If the cauchy point if outside the trust region, scale
    # the step down to hit the trust region boundary
    if norm(cache.dq_cauchy) > Δ
        dq .= cache.dq_cauchy
        fact = Δ ./ norm(cache.dq_cauchy)
        dq .*= fact
        return true, :cauchy, fact
    end

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Otherwise compute the intersection of the line segment
    # between the Cauchy and the Newton points with the trust
    # region boundary.
    
    # First compute difference between Newton and Cauchy points
    cache.q_tmp .= cache.dq_newton .- cache.dq_cauchy
    
    # then solve quadratic problem
    τ = _solve_tr_boundary!(cache.dq_cauchy, cache.q_tmp, Δ)

    # and finally compute the intersection point
    dq .= cache.dq_cauchy .+ τ .* (cache.q_tmp)

    return true, :dogleg, τ
end

# Solve for the largest τ such that ||q + τ*p||^2 = Δ^2
function _solve_tr_boundary!(dq_C, dq_N_minus_dq_C, Δ::Real)
    # compute coefficients of the quadratic equation
    a = norm(dq_N_minus_dq_C)^2
    b = 2*dot(dq_C, dq_N_minus_dq_C)
    c = norm(dq_C)^2 - Δ^2
    # compute discriminant and then return positive (largest) root
    sq_discr = sqrt(b^2 - 4*a*c)
    return max(- b + sq_discr, - b - sq_discr)/2a
end