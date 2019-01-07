# linearised operator
const _c = [1 2; 3 4]
L(t, x, y, dydt) = (dydt[1] = _c[1, 1]*x[1]*y[1] + _c[1, 2]*x[1]*y[2];
                    dydt[2] = _c[2, 1]*x[2]*y[1] + _c[2, 2]*x[2]*y[2]; dydt)

L⁺(t, x, y, dydt) = (dydt[1] = _c[1, 1]*x[1]*y[1] + _c[2, 1]*x[2]*y[2];
                     dydt[2] = _c[1, 2]*x[1]*y[1] + _c[2, 2]*x[2]*y[2]; dydt)

@testset "Operator                               " begin
    Random.seed!(0)
    M, ORDER = 100, 4
    u = StateSpaceLoop([rand(2) for i = 1:M], ORDER)
    q = PeriodicOrbit(StateSpaceLoop([rand(2) for i = 1:M], ORDER), 4)
    p = PeriodicOrbit(StateSpaceLoop([rand(2) for i = 1:M], ORDER), 5)

    op = VaPOrE._Operator(L, L⁺, q)
    a = dot(mul!(similar(u), op, p), u)
    b = dot(mul!(similar(p), op, u, ADJOINT()), p)
    @test abs(a - b) < 1e-14
end