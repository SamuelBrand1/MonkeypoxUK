using Graphs,MetaGraphs,Distributions,Plots#,GraphRecipes
using GLMakie, GraphMakie
using GraphMakie.NetworkLayout
## Affiliation generation process
#From Co-evolution of social and affiliation networks Zheleva et al 2009

@time G = stochastic_block_model(50.0, 1.0, fill(10000,10))
# function plot_this(G)
#     graphplot(G,curves = false)
# end
gay_nightclubs_london = 51
prop_lgb_london = 0.026
N_london = 8.9e6



# @time plot_this(G)

# graphplot(G)
degs = length.(G.fadjlist)
histogram(degs)