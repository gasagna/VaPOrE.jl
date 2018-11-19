# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

function solve_tr_problem!(dq::PeriodicOrbit, cache::Cache, tr_radius::Real)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Newton step first
    solve_newton!(cache)
    
    # If the Newton step is within the trust region return
    # this point, which is hopefully a good descent step
    if norm(cache.dq_newton) < tr_radius
        dq .= cache.dq_newton
        return false
    end

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # If not compute cauchy point, the minimizer along
    # the steepest descent direction
    solve_cauchy!(cache)

    # If the cauchy point if outside the trust region, scale
    # the step down to hit the trust region boundary
    if norm(cache.dq_cauchy) > tr_radius
        dq .= cache.dq_cauchy
        dq .*= tr_radius ./ norm(cache.dq_cauchy)
        return true
    end

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Otherwise compute the intersection of the line segment
    # between the Cauchy and the Newton points with the trust
    # region boundary.
    
    # First compute difference between Newton and Cauchy points
    cache.tmp .= cache.dq_newton .- cache.dq_cauchy
    
    # then solve quadratic problem
    τ = _solve_tr_boundary!(cache.dq_cauchy, cache.tmp, tr_radius)

    # and finally compute the intersection point
    dq .= cache.dq_cauchy .+ τ .* (cache.tmp)

    return true
end

# Solve for the largest τ such that ||q + τ*p||^2 = tr_radius^2
function _solve_tr_boundary!(q, p, tr_radius::Real)
    # compute coefficients of the quadratic equation
    a = norm(p)^2
    b = 2*dot(p, q)
    c = norm(q)^2 - tr_radius^2
    # compute discriminant and then return positive (largest) root
    sq_discr = sqrt(b^2 - 4*a*c)
    return max(- b + sq_discr, - b - sq_discr)/2a
end