# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

# VAriational Periodic ORbit findEr
module VaPOrE

include("output.jl")
include("options.jl")
include("statespaceloop.jl")
include("periodicorbit.jl")
include("operator.jl")
include("cache.jl")
include("dogleg.jl")
include("newton.jl")

end