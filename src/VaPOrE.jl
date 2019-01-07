# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

# VAriational Periodic ORbit findEr
module VaPOrE

# tag to trigger the adjoint calculation
struct ADJOINT end

export ADJOINT

include("output.jl")
include("options.jl")
include("statespaceloop.jl")
include("periodicorbit.jl")
include("operator.jl")
include("cache.jl")
include("dogleg.jl")
include("newton.jl")

end