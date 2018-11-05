# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

export PeriodicVector

# ~~~ PERIODIC VECTOR ~~~
struct PeriodicVector{X, N, V<:AbstractVector{X}} <: AbstractVector{X}
    data::V
    PeriodicVector(data::V) where {X, V<:AbstractVector{X}} =
        new{X, length(data), V}(data)
end

# ~ indexing interface ~
@inline Base.getindex(u::PeriodicVector{X, N}, i::Integer) where {X, N} =
    u.data[mod(i-1, N)+1]
@inline Base.setindex!(u::PeriodicVector{X, N}, val::X, i::Integer) where {X, N} =
    (u.data[mod(i-1, N)+1] = val)

# ~ linear indexing ~
Base.size(u::PeriodicVector{X, N}) where {X, N} = (N,)
Base.IndexStyle(::Type{<:PeriodicVector}) = Base.IndexLinear()

# ~ obey GMRES interface ~
Base.similar(u::PeriodicVector) = PeriodicVector(similar.(u.data))
# (a, b) = \int_0^2π a(s) \cdot b(s) ds 
# Base.dot(a::PeriodicVector{X, N}, b::PeriodicVector{X, N}) where {X, N} =
#     2π*sum(dot(a[i], b[i]) for i = 1:N)/N
# Base.norm(a::PeriodicVector) = sqrt(dot(a, a))

# # ~ broadcast the broadcast function to the underlying elements
# @generated function Base.Broadcast.broadcast!(f, u::PeriodicVector{X, N}, args::Vararg{Any, n}) where {X, N, n}
#     quote 
#         $(Expr(:meta, :inline))
#         for i = 1:$N
#             broadcast!(f, u.data[i], 
#                 $((args[j] <: PeriodicVector ? :(args[$j][i]) : :(args[$j]) for j = 1:n)...))
#         end
#         u
#     end
# end