using Graphs,MetaGraphs,Distributions,Plots# ,GraphRecipes
using GLMakie, GraphMakie
using SparseArrays,LinearAlgebra
using GraphMakie.NetworkLayout,RCall
using CSV, DataFrames

## methods for generatine power law Distributions and graph generation

function draw_pwr_law(α;x_min = 1, x_max = 2800)
    return floor(Int64,(x_min^(-α) - (x_min^(-α) - x_max^(-α))*rand())^(-1/α))
end

function create_initial_infected!(G::MetaGraph,ϕ;α = 0.8,time_period = 365.25)
    add_vertex!(G)#Add infected person
    old_size_graph = nv(G) #index of node just added
    new_inf_list = push!(get_prop(G,:inf_nodes),old_size_graph)
    set_prop!(G,:inf_nodes,new_inf_list) #Add new infected to list of infected
    set_props!(G,old_size_graph,Dict(:inf_status => :I,:inf_time => rand(1:20))) #Properties of infected person
    contacts = draw_pwr_law(α)#Generate contacts from power law distribution
    add_vertices!(G,contacts)
    for i in (old_size_graph+1):nv(G)
        add_edge!(G,old_size_graph,i) #edges to new contacts
        set_prop!(G,old_size_graph,i,:time_left,rand()*time_period) #Time to contact
        set_props!(G,i,Dict(:inf_status => :S)) #Properties of infected person
        for j = (i+1):nv(G) #Possible clustering in graph
            if rand() < ϕ
                add_edge!(G,i,j) # Only need to add the time left to contact if node goes infectious 
            end
        end
    end
    return nothing
end

function infectious_periods!(G;infectious_duration = 21)
    for i in get_prop(G,:inf_nodes)
        set_prop!(G,i,:inf_time, get_prop(G,i,:inf_time) + 1)# extra day since infection
        if get_prop(G,i,:inf_time) == infectious_duration            
            set_prop!(G,i,:inf_status,:R) #Recovery
            set_prop!(G,:inf_nodes,filter(v -> v != i,get_prop(G,:inf_nodes))) #Remove from infectious list
        end
    end
end

function infect_node!(G,inf_node,σ,ϕ;α = 0.8,time_period = 365.25,max_size = 10_000)
    new_inf_list = push!(get_prop(G,:inf_nodes),inf_node) #Add infected node to list
    set_prop!(G,:inf_nodes,new_inf_list) #Add new infected to list of infected
    set_props!(G,inf_node,Dict(:inf_status => :I,:inf_time => 1)) #Properties of infected person    
    #Activate any contacts already generated
    for j in neighbors(G,inf_node)
        if !has_prop(G,inf_node,j,:time_left)
            set_prop!(G,inf_node,j,:time_left,rand()*time_period)
        end
    end
    contacts = max(draw_pwr_law(α-σ) - length(neighbors(G,inf_node)),0)#Generate contacts from power law distribution with bias due to assortativity less the contacts already existing
    
    old_size_graph = nv(G)
    if old_size_graph < max_size # Grow the graph
        add_vertices!(G,contacts) #Add contacts as susceptible vertices
        for i in (old_size_graph+1):nv(G)
            add_edge!(G,inf_node,i) #edges to new contacts
            set_prop!(G,inf_node,i,:time_left,rand()*time_period) #Time to contact
            set_prop!(G,i,:inf_status, :S) #Properties of infected person
            for j = (i+1):nv(G) #Possible clustering in graph
                if rand() < ϕ
                    add_edge!(G,i,j) # Only need to add the time left to contact if node goes infectious 
                end
            end
        end
    else #Add links
        for c in 1:contacts
            contacted_node = rand(filter(v -> v != inf_node,1:old_size_graph))
            if get_prop(G,contacted_node,:inf_status) == :S
                add_edge!(G,inf_node,contacted_node) #edges to new contacts
                set_prop!(G,inf_node,contacted_node,:time_left,rand()*time_period) #Time to contact
            end
        end
    end
end

function infections!(G,infectivity_profile,σ,ϕ;α = 0.8,time_period = 365.25,max_size = 100_000)
    for i in get_prop(G,:inf_nodes)
        for j in neighbors(G,i)
            τ = get_prop(G,i,j,:time_left) - 1.0 #Decrease time to event
            set_prop!(G,i,j,:time_left,τ)
            time_since_inf = get_prop(G,i,:inf_time)
            if τ < 0 && get_prop(G,j,:inf_status) == :S && rand() < infectivity_profile[time_since_inf]
                infect_node!(G,j,σ,ϕ;α = α,time_period = time_period) #infection event
            end
        end
    end
end


baseline_infectivity_profile=[0 0 0 0.2 0.2 0.2 0.3 0.5 0.8 1 1 1 1 1 1 1 1 1 0.8 0.5 0][:]

α₀ = 0.8
## Create basic graph
G = MetaGraph()
set_prop!(G,:inf_nodes,Int64[])
for init = 1:100
    create_initial_infected!(G,0.1)
end
##


# infect_node!(G,2,1.0,0.1)
# infections!(G,baseline_infectivity_profile,1.0,0.1;α = 0.8,time_period = 365.25)
##

infections!(G,0.1.*baseline_infectivity_profile,1.0,0.5;α = 0.8,time_period = 365.25)
infectious_periods!(G)
length(get_prop(G,:inf_nodes))
