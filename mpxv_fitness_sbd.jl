## Idea is to have both fitness and SBM effects in sexual contact

using Graphs,MetaGraphs,Distributions,Plots# ,GraphRecipes
using GLMakie, GraphMakie
using SparseArrays,LinearAlgebra
using GraphMakie.NetworkLayout,RCall
using CSV, DataFrames,SparseArrays

## methods for generatine power law Distributions and graph generation

function draw_pwr_law(α;x_min = 1, x_max = 2800)
    return floor(Int64,(x_min^(-α) - (x_min^(-α) - x_max^(-α))*rand())^(-1/α))
end
## Basic parameters

baseline_infectivity_profile=[0 0 0 0.2 0.2 0.2 0.3 0.5 0.8 1 1 1 1 1 1 1 1 1 0.8 0.5 0][:]
α₀ = 0.8
N::Integer = 200_000
n_c = 50 # Number of communities
p_in = 0.9 #proportion of sexual contacts within group
##

function create_initial_graph(N,n_c;α = 0.8)
    G = MetaGraph(0)
    for i = 1:N
        add_vertex!(G)
        set_props!(G,i,Dict(:rate => draw_pwr_law(α) / 2, :community => rand(1:n_c), :inf_status => :S, :inf_time => -1))
    end
    return G
end


