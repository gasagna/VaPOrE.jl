using VaPOrE
using Base.Test

# Problem from section 4 in Viswanath 2001
# ẋ = -y + μx(1 - √(x² + y²))
# ẏ =  x + μy(1 - √(x² + y²))
# The solution is the limit cycle
# x(t) = cos(t)
# y(t) = sin(t)
# with period T=2π and angular frequency ω=1
# The velocities are:
# ẋ(t) = -sin(t)
# ẏ(t) =  cos(t)

struct System
    μ::Float64
end

# right hand side
function (s::System)(t, x, dxdt)
    x_, y_, μ = x[1], x[2], s.μ
    r = sqrt(x_^2 + y_^2)
    @inbounds dxdt[1] = - y_ + μ*x_*(1 - r)
    @inbounds dxdt[2] =   x_ + μ*y_*(1 - r)
    return dxdt
end

# linearised operator
struct SystemLinear{ISADJOINT}
    μ::Float64
    J::Matrix{Float64}
    SystemLinear{ISADJOINT}(μ::Real) where {ISADJOINT} = new{ISADJOINT}(μ, zeros(2, 2))
end

function (s::SystemLinear{ISADJOINT})(t, x, v, out) where {ISADJOINT}
    x_, y_, μ = x[1], x[2], s.μ
    r = sqrt(x_^2 + y_^2)
    @inbounds s.J[1, 1] = μ*(1 - r - x_^2/r)
    @inbounds s.J[1, 2] = -1 - μ*x_*y_/r
    @inbounds s.J[2, 1] =  1 - μ*x_*y_/r
    @inbounds s.J[2, 2] = μ*(1 - r - y_^2/r)
    return ISADJOINT ? At_mul_B!(out, s.J, v) : A_mul_B!(out, s.J, v)
end

# define systems
μ = 1.0
F = System(μ)
D = SystemLinear{false}(μ)
A = SystemLinear{true}(μ);

# define initial guess, a slightly perturbed orbit
M = 50
ts = linspace(0, 2π, M+1)[1:end-1]
q = PeriodicOrbit(StateSpaceLoop([1.7*[cos(t), sin(t)] for t in ts], 8), 5);

# search
search!(q, F, D, A, Options(maxiter=15, r_norm_tol=1e-14))

# solution is a loop of unit radius and with \omega = 1
@test maximum( norm.(q.u) - 1 ) < 1e-10
@test abs(q.ds[1] - 1 ) < 1e-10