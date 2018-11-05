using Base.Test
using VaPOrE

# ~~~ BEGIN ~ needed for broadcast testing ~~~
struct Foo{T, V<:AbstractVector{T}} 
    data::V
end

Foo(data::AbstractVector) = Foo{eltype(data), typeof(data)}(data)

@generated function Base.Broadcast.broadcast!(f, u::Foo, args::Vararg{Any, n}) where {n}
    quote 
        $(Expr(:meta, :inline))
        broadcast!(f, unsafe_get(u), map(unsafe_get, args)...)
        u
    end
end

Base.unsafe_get(u::Foo) = u.data
# ~~~ END ~ needed for broadcast testing ~~~

@testset "PeriodicVector                         " begin
    @testset "periodic indexing                         " begin
        u = PeriodicVector([[1], [2], [3]])
        @test u[-1] == [2]
        @test u[ 0] == [3]
        @test u[ 1] == [1]
        @test u[ 2] == [2]
        @test u[ 3] == [3]
        @test u[ 4] == [1]
        @test u[ 5] == [2]

        u[5] .= [4]
        @test u[2] == [4]
    end

    @testset "norm, dot                          " begin
        t = linspace(0, 2π, 15)[1:14]

        # integral of cos(t)^2 = π
        u = PeriodicVector([[cos(ti)] for ti in t])
        @test abs(dot(u, u) - π) < 1e-16
        @test abs(norm(u) - sqrt(π)) < 1e-16

        # integral of cos(2t)sin(x) = 0
        u = PeriodicVector([[cos(2ti)] for ti in t])
        v = PeriodicVector([[sin(ti)]  for ti in t])
        @test abs(dot(u, v) - 0) < 1e-16
    end
end