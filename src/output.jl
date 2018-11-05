# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

# Display headers
const _header = "+------+----------+-----------+-----------+----------+-----------+----------+----------+\n"*
                "| iter |   |δx|   |    δT     |    δv     |    T     |     v     |    |e|   |     λ    |\n"*
                "+------+----------+-----------+-----------+----------+-----------+----------+----------+\n"

display_header() = (print(_header); flush(STDOUT))

function display_status(iter, δx_norm, δT, δv, T, v, r_norm, λ)
    str = @sprintf "|%4d  | %5.2e | %+5.2e | %+5.2e | %5.2e | %+5.2e | %5.2e | %5.2e |" iter δx_norm δT δv T v r_norm λ
    println(str)
    flush(STDOUT) 
end