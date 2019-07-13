@testset "StateSpaceLoop                         " begin
    @testset "constructor                        " begin
        @test_throws ArgumentError StateSpaceLoop([[1], [2], [3]], 1)
    end
    @testset "periodic indexing                  " begin
        u = StateSpaceLoop([[1], [2], [3]], 2)
        @test order(u) == 2
        @test u[-1] == [2]  
        @test u[ 0] == [3]
        @test u[ 1] == [1]
        @test u[ 2] == [2]
        @test u[ 3] == [3]
        @test u[ 4] == [1]
        u[4] .= [4]
        @test u[1] == [4]
    end
    @testset "prolong & restrict                 " begin
        v = StateSpaceLoop([[1.0], [2.0], [3.0]], 2)
        u = prolong(v)
        @test u[ 1] == [1]
        @test u[ 2] == [1.5]
        @test u[ 3] == [2]
        @test u[ 4] == [2.5]
        @test u[ 5] == [3]
        @test u[ 6] == [2]
        w = restrict(u)
        @test w == v
        @test_throws ArgumentError restrict(v)
    end
    @testset "norm, dot                          " begin
        u = StateSpaceLoop([[1], [2], [3]], 2)
        v = StateSpaceLoop([[2], [0], [1]], 2)
        @test norm(u)   == sqrt( (1^1 + 2^2 + 3^2)/3 )
        @test dot(u, v) ==       (1*2 + 2*0 + 3*1)/3
    end
    @testset "broadcast                          " begin
        u = StateSpaceLoop([[1], [2], [3]], 2)
        v = StateSpaceLoop([[2], [0], [1]], 2)
        w = similar(u)
        w .= u .+ 2.0.*v
        @test w[1] == [5]
    end
    @testset "broadcast allocations              " begin
        u = StateSpaceLoop([randn(100) for i = 1:500], 2)
        v = StateSpaceLoop([randn(100) for i = 1:500], 2)
        w = similar(u)
        foo(u, v, w) = (@allocated w .= u .+ 2.0.*v .- 1.0.*u)
        @test foo(u, v, w) == 0
    end
    @testset "loop derivative                    " begin
        # differentiate cos(t)
        for (order, tol) in zip((2,    4,     6,     8,      10),
                                (0.17, 0.0331, 0.0072, 0.0016, 0.00036))
            for M = [10, 15, 20, 25, 30, 35]
                t = range(0, stop=2π, length=M)[1:end-1]
                u    = StateSpaceLoop([[cos(ti)] for ti in t], order)
                duds = dds!(u, 1, similar(u[1]))
                ϵ = abs(duds[1] + sin(t[1])) / (t[2]-t[1])^(order+1)
                @test ϵ < tol
            end
        end
    end
end