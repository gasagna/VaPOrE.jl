# ------------------------------------------------------------------- #
# Copyright 2017-2019, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
using Printf

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRUST REGION ALGORITHM
#
# Display header
const _header_tr = "+------+--------+-----------+-----------+------------+-----------+-----------+------------+----------------+\n"*
                   "| iter | which  |   step    |  ||res||  |      ρ     |     Δ     |  ||dq||   |     dω     |        ω       |\n"*
                   "+------+--------+-----------+-----------+------------+-----------+-----------+------------+----------------+\n"

display_header_tr() = (print(_header_tr); flush(stdout))

function display_status_tr(iter, which, step, res_norm, ρ, Δ, dq_norm, dω, ω)
    str = @sprintf "|%4d  | %s | %5.3e | %5.3e | %+5.3e | %5.3e | %5.3e | %+5.3e | %9.8e |" iter which step res_norm ρ Δ dq_norm dω ω
    println(str)
    flush(stdout) 
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LINE SEARCH ALGORITHM
#
# Display header
const _header_ls = "+------+----------+-----------+-----------+----------+-----------+----------+----------+\n"*
                   "| iter |   |δx|   |    δT     |    δv     |    T     |     v     |    |e|   |     λ    |\n"*
                   "+------+----------+-----------+-----------+----------+-----------+----------+----------+\n"

display_header_ls() = (print(_header_ls); flush(stdout))

function display_status_ls(iter, δx_norm, δT, δv, T, v, r_norm, λ)
    str = @sprintf "|%4d  | %5.2e | %+5.2e | %+5.2e | %5.2e | %+5.2e | %5.2e | %5.2e |" iter δx_norm δT δv T v r_norm λ
    println(str)
    flush(stdout) 
end