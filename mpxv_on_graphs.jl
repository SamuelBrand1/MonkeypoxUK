using Graphs,MetaGraphs,Distributions,Plots# ,GraphRecipes
using GLMakie, GraphMakie
using SparseArrays,LinearAlgebra
using GraphMakie.NetworkLayout,RCall
using CSV, DataFrames

## Network building parameters

prop_lgb = 0.026
n_lgb = 100
@rput n_lgb

## People
infectivity_profile=[0 0 0 0.2 0.2 0.2 0.3 0.5 0.8 1 1 1 1 1 1 1 1 1 0.8 0.5 0][:]

Base.@kwdef mutable struct Person
    age::Int64
    disease_status::Char = 'S'
    days_since_inf::Int64 = 0
    household_id::Int64 = -1 #-1 means unassigned
    work_id::Int64 = -1 
    msm::Bool = true
end

natsal_data = CSV.File("NATSAL3_MSM_header.csv") |> DataFrame
p_choose = normalize(natsal_data.total_wt,1)
hh_distrib = normalize!([7452,9936,4416,4140,1104,552].*1e3,1)

population = [Person(age = natsal_data.dage[rand(Categorical(p_choose))]) for i = 1:n_lgb]


## sexual contact matrix from simdynet
R"""
    library(simdynet)
    Sys.setenv('R_MAX_VSIZE'=320000000000)
    s <- sim_static_sn(N=n_lgb,gamma = 1.8)

"""
sex_cnt_net = @rget s
cnt_mat = sparse(sex_cnt_net[:res] .+ sex_cnt_net[:res]')
G = Graph(cnt_mat)
cnt_G = MetaGraph(G)
# graphplot(G_sexct)
#Set the base graph as sexual contacts
for e in edges(cnt_G)
    set_prop!(cnt_G,e,:type,:sexual)
end
## Generate household contacts
function assign_household_ids!(population,G,prob_ingroup = prop_lgb)
    curr_hh_id = 1
    for person in population
        if person.household_id == -1 && person.msm
            n_hh = rand(Categorical(hh_distrib)) - 1 #Number of people in household
            new_people = rand(Binomial(n_hh, 1 - prob_ingroup)) # New non-lgb people
            add_vertices!(G,new_people)
            curr_people = n_hh - new_people #Household links to people already in population
            person.household_id = curr_hh_id
            for np in 1:new_people
                sampleage = natsal_data.dage[rand(Categorical(p_choose))]
                push!(population,Person(age = sampleage,household_id = curr_hh_id,msm = false))#New person in current household
            end
            for np in 1:curr_people
                f = findall([person.household_id == -1 for person in population]) #Index currently unassigned people in population
                choose = rand(f) #Select an unassigned person already in the population
                population[choose].household_id = curr_hh_id
            end
            curr_hh_id +=1 
        end
    end
    return nothing
end

function assign_work_ids!(population,G,prob_ingroup = prop_lgb,mean_wk_cnts = 7)
    curr_wk_id = 1
    for person in population
        if person.work_id == -1 && person.msm
            n_wk = rand(Poisson(mean_wk_cnts)) #Number of people in workplace
            new_people = rand(Binomial(n_wk, 1 - prob_ingroup)) # New non-lgb people
            add_vertices!(G,new_people)
            curr_people = n_wk - new_people #Household links to people already in population
            person.work_id = curr_wk_id
            for np in 1:new_people
                sampleage = natsal_data.dage[rand(Categorical(p_choose))]
                push!(population,Person(age = sampleage,work_id = curr_wk_id,msm = false))#New person in current workplace
            end
            for np in 1:curr_people
                f = findall([person.work_id == -1 for person in population]) #Index currently unassigned people in population
                choose = rand(f) #Select an unassigned person already in the population
                population[choose].work_id = curr_wk_id
            end
            curr_wk_id +=1 
        end
    end
    return nothing
end

function add_household_contacts!(population,G)
    for (i,person) in enumerate(population)
        hh_id = person.household_id
        for j in (i+1):length(population)
            population[j].household_id == hh_id && add_edge!(G,i,j,:type,:hh)
        end
    end
    return nothing
end

function add_work_contacts!(population,G)
    for (i,person) in enumerate(population)
        wk_id = person.work_id
        for j in (i+1):length(population)
            population[j].work_id == wk_id && add_edge!(G,i,j,:type,:work)
        end
    end
    return nothing
end
population
cnt_G
assign_household_ids!(population,cnt_G)
population
cnt_G
assign_work_ids!(population,cnt_G)
population
cnt_G
add_household_contacts!(population,cnt_G)
add_work_contacts!(population,cnt_G)

@time graphplot(cnt_G)

## Add work contacts

g = watts_strogatz(1000,7,0.)
@time graphplot(g,title = "work")





# @time G = stochastic_block_model(50.0, 1.0, fill(10000,10))
# # function plot_this(G)
# #     graphplot(G,curves = false)
# # end
# gay_nightclubs_london = 51
# prop_lgb_london = 0.026
# N_london = 8.9e6
# G = watts_strogatz(100,7,0.9)
# graphplot(G)

# # @time plot_this(G)

# # graphplot(G)
# degs = length.(G.fadjlist)
# histogram(degs)


function draw_pwr_law(α;x_min = 1.0, x_max = 2800.0)
    return (x_min^(-α) - (x_min^(-α) - x_max^(-α))*rand())^(-1/α)
end



Base.@kwdef mutable struct InfPerson
    episode_phase::Symbol = :incubation
    incubation_stage::Integer = 1
    will_be_detected::Bool
    contact_rate::Float64
end

Base.@kwdef mutable struct Group
    highfreq::Bool
    pop_size::Integer
    contact_rates::Vector{Float64} = [0.0]
    susceptible::Vector{Bool} = [true]
    number_traced_contacts::Vector{Int64} = [1]
    infecteds::Union{Vector{InfPerson},Vector{Int64}}
end

Base.@kwdef mutable struct Population
    groups::Vector{Group}
end

function create_group(pop_size,p_detect,α,prop_init_inf,threshold;highfreq = true,x_max = 2800.0)
    if highfreq
        init_infs = rand(pop_size) .< prop_init_inf
        ctrs = [draw_pwr_law(α;x_min = threshold, x_max = x_max) for n = 1:pop_size]
        sus = trues(pop_size)
        sus[init_infs] .= false
        init_inf_array = [InfPerson(incubation_stage = rand(1:5),will_be_detected = rand() < p_detect,contact_rate = ctr)  for ctr in ctrs[init_infs]]
        group = Group(highfreq = true,
                    pop_size = pop_size,
                    contact_rates = ctrs,
                    susceptible = sus,
                    number_traced_contacts = zeros(Int64,pop_size),
                    infecteds = init_inf_array)
        return group
    else
        init_inf_array = zeros(Int64,6)
        group = Group(highfreq = false,
                    pop_size = pop_size,
                    contact_rates = zeros(pop_size),
                    susceptible = trues(pop_size),
                    number_traced_contacts = zeros(Int64,pop_size),
                    infecteds = init_inf_array)
        return group
    end
end

function create_population(n_groups,α₀,p_detect,α,prop_init_inf,N_msm::Integer,N_nonmsm::Integer;threshold = 30.0,x_max = 2800.0)
    groups = Vector{Group}(undef,n_groups+1)
    prob_above_threshold = 1 - ((1 - threshold^(-α))/(1 - x_max^(-α)))
    N_high_contact = rand(Binomial(N_msm,prob_above_threshold))

    pop_sizes = rand(DirichletMultinomial(N_high_contact,α₀ * ones(n_groups)))
    for n = 1:n_groups
        group = create_group(pop_sizes[n],p_detect,α,prop_init_inf,threshold)
        groups[n] = group
    end
    group_nonmsm = create_group(N_nonmsm + N_msm - N_high_contact,p_detect,α,prop_init_inf,threshold;highfreq = false)
    groups[end] = group_nonmsm
    return groups
end

@time group = create_group(6000,0.5,0.8,0.0001,30.0)
@code_warntype create_group(600,0.5,0.8,0.0001,30.0)
ProfileView.@profview create_group(6000,0.5,0.8,0.0001,30.0)
@profile create_group(6000,0.5,0.8,0.0001,30.0)
function testfunct()
    group = create_group(100_000,0.5,0.8,0.0001,30.0)
end
group_nonmsm = create_group(100_000,0.5,0.8,0.0001;msm = false)

@code_warntype create_population(1,3.0,0.5,0.8,0.000001,1_700_000,Int64(66e6))
@time create_population(1,3.0,0.5,0.8,0.000001,1_700_000,Int64(66e6))
@profile create_population(1,3.0,0.5,0.8,0.000001,1_700_000,Int64(66e6)) 


@time pop_sizes = rand(DirichletMultinomial(1_700_000,3.0 * ones(10)))
@time v = Vector{Group}(undef,10)

function ranthis()
    ctrs = [draw_pwr_law(0.8) for n = 1:15000]./365.25
end
function ranthis!(ctrs)
    for i = 1:15000
        ctrs[i] = draw_pwr_law(0.8)/365.25
    end
end

@time ctrs = ranthis()

@time ranthis!(ctrs)