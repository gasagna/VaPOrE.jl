@testset "PeriodicOrbit                          " begin
    @testset "norm, dot                          " begin
        u = PeriodicOrbit(StateSpaceLoop([[1], [2], [3]], 2), 2.0)
        v = PeriodicOrbit(StateSpaceLoop([[4], [5], [6]], 2), 3.0)

        @test norm(u)   == sqrt( (1^1 + 2^2 + 3^2)/3 + 2.0^2 )
        @test dot(u, v) ==       (1*4 + 2*5 + 3*6)/3 + 2*3
    end
    @testset "broadcast                          " begin
        u = PeriodicOrbit(StateSpaceLoop([[1], [2], [3]], 2), 2.0)
        v = PeriodicOrbit(StateSpaceLoop([[4], [5], [6]], 2), 3.0)
        w = similar(u)
        w .= u .+ 2.0.*v
        @test w.u[1] == [9]
        @test w.u[2] == [12]
        @test w.u[3] == [15]
        @test w.ds   == (8.0, )
    end
    @testset "broadcast allocations              " begin
        u = PeriodicOrbit(StateSpaceLoop([[1], [2], [3]], 2), 2.0)
        v = PeriodicOrbit(StateSpaceLoop([[4], [5], [6]], 2), 3.0)
        w = similar(u)
        foo(u, v, w) = (@allocated w .= u .+ 2.0.*v)
        # @test foo(u, v, w) == 0
    end
end