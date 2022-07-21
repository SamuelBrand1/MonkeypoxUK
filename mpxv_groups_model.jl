## Idea is to have both fitness and SBM effects in sexual contact

using Distributions,Plots,StatsBase
using SparseArrays,LinearAlgebra
using CSV, DataFrames,SparseArrays

## Group size
N_uk = 67.22e6 
prop_ovr18 = 0.787
prop_msm = 0.034 #https://www.ons.gov.uk/peoplepopulationandcommunity/culturalidentity/sexuality/bulletins/sexualidentityuk/2020
prop_sexual_active = 1 - 0.154 #A dynamic power-law sexual network model of gonorrhoea outbreaks

N = round(Int64,N_uk*prop_ovr18*prop_msm*prop_sexual_active) #~1.5m

## Incubation period

# Using  this  best-fitting  distribution,  the  mean  incuba-tion period was estimated to be 8.5 days
#  (95% credible intervals (CrI): 6.6–10.9 days), 
# with the 5th percentile of 4.2 days and the 95th percentile of 17.3 days (Table 2)
##
d_incubation = Gamma(5,8.5/5)
mean(d_incubation),quantile(d_incubation,0.05),
    quantile(d_incubation,0.95)

d_incubation = LogNormal(2.09,0.46)#<--- from running the code and using posterior mean
mean(d_incubation),quantile(d_incubation,0.05),
    quantile(d_incubation,0.95)
d_infectious = Gamma(1,7/1)
p_inc = [cdf(d_incubation,t) - cdf(d_incubation,t-1) for t = 1:60]
Q_inf = [ccdf(d_infectious,t) for t = 1:60]
p_inf = [ sum(p_inc[1:(t-1)].*Q_inf[(t-1):-1:1]) for t = 1:60 ]
w = normalize(p_inf,1)
bar(p_inf)
r = log(3.5)/55

R0 = 1/sum([exp(-r*t)*w[t] for t = 1:60])


##
# x = rand(DirichletMultinomial(N,100 * ones(10)))
# bar(x)
# function set_up_initial_conditions()

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