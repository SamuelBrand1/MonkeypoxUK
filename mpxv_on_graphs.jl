using Graphs,MetaGraphs,Distributions,Plots #,GraphRecipes
using GLMakie, GraphMakie
using GraphMakie.NetworkLayout,RCall

## Network building parameters

prop_lgb = 0.026
n_lgb = 1000

## sexual contact matrix from simdynet
R"""
    library(simdynet)
    s <- sim_static_sn(N=100,gamma = 1.8) 
"""
sex_cnt_net = @rget s
cnt_mat = sex_cnt_net[:res] .+ sex_cnt_net[:res]'
G_sexct = Graph(cnt_mat)
graphplot(G_sexct)

##
@time G = stochastic_block_model(50.0, 1.0, fill(10000,10))
# function plot_this(G)
#     graphplot(G,curves = false)
# end
gay_nightclubs_london = 51
prop_lgb_london = 0.026
N_london = 8.9e6
G = watts_strogatz(100,7,0.9)
graphplot(G)

# @time plot_this(G)

# graphplot(G)
degs = length.(G.fadjlist)
histogram(degs)