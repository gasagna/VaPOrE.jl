# linearised operator
const _c = [1 2; 3 4]
L(t, x, y, dydt) = (dydt[1] = _c[1, 1]*x[1]*y[1] + _c[1, 2]*x[1]*y[2];
                    dydt[2] = _c[2, 1]*x[2]*y[1] + _c[2, 2]*x[2]*y[2]; dydt)

L⁺(t, x, y, dydt) = (dydt[1] = _c[1, 1]*x[1]*y[1] + _c[2, 1]*x[2]*y[2];
                     dydt[2] = _c[1, 2]*x[1]*y[1] + _c[2, 2]*x[2]*y[2]; dydt)

@testset "Operator                          " begin
    srand(0)
    M, ORDER = 100, 4
    u = PeriodicTrajectory([rand(2) for i = 1:M], ORDER)
    q = PeriodicOrbit(PeriodicTrajectory([rand(2) for i = 1:M], ORDER), 4, 0)
    p = PeriodicOrbit(PeriodicTrajectory([rand(2) for i = 1:M], ORDER), 5, 0)

    op = VaPOrE._Operator(L, L⁺, q)
    a = dot( A_mul_B!(similar(u), op, p), u)
    b = dot(At_mul_B!(similar(p), op, u), p)
    @test abs(a - b) < 1e-14
end