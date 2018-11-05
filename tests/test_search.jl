using Base.Test
using KS
using VaPOrE
using IMEXRKCB

Gx(qi::FTField{n, L}, tmp::FTField{n, L}) where {n, L} = 
    (tmp .= qi.*im.*(0:n).*2π./L; tmp)

@testset "search " begin

    # problem parameters
    L  = 22
    n  = 32
    U  = 0.0
    Δt = 0.5

    # rhs of equation
    F = KSEq(n, U, L)

    # explicit and implicit terms
    imTerm, exTerm = imex(F)

    # initial condition
    uk = FTField(n, L); uk[1] = 0.1; uk[2] = 0.2; uk[3] = 0.3

    # create scheme 
    scheme = IMEXRKScheme(IMEXRK4R3R(IMEXRKCB4, false), uk)

    # and integrator
    G = integrator(exTerm, imTerm, scheme, Δt)

    # monitor solution
    mon = Monitor(uk, copy)

    # advance a little bit
    G(uk, (0, 100))

    # now get a guess
    G(uk, (0, 32), mon)

    # create PO object
    q = PeriodicOrbit(mon.xs, 2π/32, 0)

    # get linearised operator
    FL = LinearisedKSEq(n, U, L)

    # search
    nksearch!(F, FL, Gx, q)

end