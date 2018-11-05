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

@testset "PeriodicOrbit                          " begin
    @testset "periodic indexing                         " begin
        u = PeriodicOrbit([[1], [2], [3]], 1, 5)
        @test u[-1] == [2]
        @test u[ 0] == [3]
        @test u[ 1] == [1]
        @test u[ 2] == [2]
        @test u[ 3] == [3]
        @test u[ 4] == [1]
        u[4] .= [4]
        @test u[1] == [4]
    end
    @testset "norm, dot                          " begin
        u = PeriodicOrbit([[1], [2], [3]], 4, 5)
        v = PeriodicOrbit([[2], [0], [1]], 2, 3)
        @test norm(u)   == sqrt(2π*(1^1 + 2^2 + 3^2)/3 + 4^2 + 5^2)
        @test dot(u, v) ==      2π*(1*2 + 2*0 + 3*1)/3 + 4*2 + 5*3
    end
    @testset "broadcast                          " begin
        u = PeriodicOrbit([[1], [2], [3]], 4, 5)
        v = PeriodicOrbit([[2], [0], [1]], 2, 3)
        w = similar(u)
        w .= u .+ 2.*v
        @test w[1] == [5]
        @test w.v  == 11
        @test w.ω  == 8
    end
    @testset "broadcast allocations              " begin
        u = PeriodicOrbit([[1], [2], [3]], 4, 5)
        v = PeriodicOrbit([[2], [0], [1]], 2, 3)
        w = similar(u)
        foo(u, v, w) = (@allocated w .= u .+ 2.*v)
        @test foo(u, v, w) == 0
    end
    @testset "broadcast or objects                    " begin
        u = PeriodicOrbit([Foo([1]), Foo([2]), Foo([3])], 4, 5)
        v = PeriodicOrbit([Foo([2]), Foo([0]), Foo([1])], 2, 3)
        u .+= v
        @test u.v == 8
        @test u.ω == 6
        @test u[1].data == Foo([3]).data # compare the data here
    end
    @testset "loop derivative                    " begin
        # differentiate cos(t)
        t = linspace(0, 2π, 100)[1:99]
        u = PeriodicOrbit([[cos(ti)] for ti in t], 1, 0)
        v = dds!(u, similar(u))
        for i = 1:99
            @test abs(v[i][1] + sin(t[i])) < 1e-6
        end
    end
end 