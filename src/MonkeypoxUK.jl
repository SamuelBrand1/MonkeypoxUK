module MonkeypoxUK

using Distributions, StatsBase
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq
using Roots

include("utils.jl")
include("dynamics.jl");

end # module MonkeypoxUK
