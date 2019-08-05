using FFTW

@testset "poincare                          " begin
    for N in [20, 21]
        for k = 0:10
            t = range(0, stop=2π, length=N+1)[1:N]
            fun(t) = cos(k*t)
            ps = fun.(t)
            p̂s = rfft(ps)
            ts = range(0, stop=2π, length=11)[1:10]
            for t in ts
                val = perinterp(length(ps), p̂s, t)
                @test abs(fun(t) - val) < 1e-14 
            end
        end
    end
end