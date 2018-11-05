# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

# VAriational Periodic ORbit findEr
module VaPOrE

include("output.jl")
include("options.jl")
include("fd.jl")
include("periodicorbit.jl")
include("sensitivity.jl")
include("nrsystem.jl")
include("linesearch.jl")
include("newton.jl")

end