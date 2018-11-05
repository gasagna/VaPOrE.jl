# ------------------------------------------------------------------- #
# Copyright 2017-2018, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #

macro checkorder(order)
    quote
        $(esc(order)) in (2, 4, 6, 8, 10) ||
            throw(ArgumentError("Error must be in (2, 4, 6, 8, 10)"))
    end
end

const _coeffs = Dict{Int, Vector{Float64}}()

_coeffs[2]  = Float64[1]./2
_coeffs[4]  = Float64[8, -1]./12
_coeffs[6]  = Float64[45, -9, +1]./60
_coeffs[8]  = Float64[672, -168, +32, -3]./840
_coeffs[10] = Float64[2100, -600, +150, -25, +2]./2520