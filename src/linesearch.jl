# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

function calc_r_norm(F,
                     D,
                     q::PO,
                    dq::PO,
                     λ::Real,
                     order::Int,
                     p::PO, dpds::PO,
                     dpdx::PO, f::PO) where {PO <: PeriodicOrbit}
    
    # compute pertubed trajectory
    p .= q .+ λ.*dq

    # and derivatives wrt to s and x, and the vector field too
    for i = 1:length(p)
        dds!(p, i, dpds[i], order)
        D isa Void || D(dpdx[i], p[i])
        F(0.0, p[i], f[i])
    end

    # calc residual, and overwrite f
    f .= p.ω .* dpds .- f
    D isa Void || (f .+= p.v .* dpdx)

    return mean(x->norm(x)^2, f)
end

function linesearch(F,
                    D,
                    q::PO,
                   dq::PO,
                    order::Int,
                    p::PO, dpds::PO,
                    dpdx::PO, f::PO, opts::Options) where {PO <: PeriodicOrbit}
    # current error
    r_norm_0 = calc_r_norm(F, D, q, dq, 0, order, p, dpds, dpdx, f)

    # start with full Newton step
    λ = 1.0

    # initialize this variable
    r_norm_λ = λ*r_norm_0

    for iter = 1:opts.ls_maxiter
        # calc new residual
        r_norm_λ = calc_r_norm(F, D, q, dq, λ, order, p, dpds, dpdx, f)
        
        # accept any reduction of error
        r_norm_λ < r_norm_0 && return λ, r_norm_λ, true
        
        # ~ otherwise attempt with shorter step ~
        λ *= opts.ls_rho
    end 

    return 1.0, r_norm_0, false
end