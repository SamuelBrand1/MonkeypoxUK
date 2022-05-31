using LinearAlgebra, Roots, StatsPlots, CSV, DataFrames, Distributions, QuadGK, DataInterpolations

## Grab the data

natsal_data = CSV.File("NATSAL3_MSM_header.csv") |> DataFrame
p_choose = normalize(natsal_data.total_wt,1)

##Find the average (population weighted) squared new contacts per day, average new contacts per contacts per day
#and their ratio

av_new_contact_sq = sum(p_choose .* (natsal_data.snonewp./365.25).^2)
av_new_contact = sum(p_choose .* natsal_data.snonewp./365.25)
ratio = av_new_contact_sq / av_new_contact

## Define the infectivity profile and a linear interpolation useful for numerical quad.

Infectivity_Profile=[0 0 0 0.2 0.2 0.2 0.3 0.5 0.8 1 1 1 1 1 1 1 1 1 0.8 0.5 0 0 0 0][:]
infectivity = LinearInterpolation(Infectivity_Profile,0:23)

## Helper functions
"""
    function mgf_infectivity(r,infectivity)

Calculate M`(-r) = ∫exp(-rt) ι(t) dt`, the Laplace transform of the infectivity profile ι,
using numerical quad.         
"""
function mgf_infectivity(r,infectivity)
    integral, err = quadgk(t -> exp(-r*t) * infectivity(t), 0, 22)
    return integral
end

"""
    function mpxv_R0(prob_per_contact,ratio,infectivity)

Calculate `R₀ = p (⟨X²⟩ / ⟨X⟩) M(0)`, where `p` is the probability of infection per
contact and X is a rand. var. of daily rate of contacts for a person. These are assumed to be
matched contacts (e.g. sexual contact).
"""
function mpxv_R0(prob_per_contact,ratio,infectivity)
    integral, err = quadgk(t -> infectivity(t), 0, 22)
    prob_per_contact * ratio * mgf_infectivity(0,infectivity)
end

"""
    function diff_r(r,prob_per_contact,ratio,infectivity)

Helper function for root finding for calculating the exponential growth rate.        
"""
function diff_r(r,prob_per_contact,ratio,infectivity)
    mgf_infectivity(r,infectivity) * prob_per_contact * ratio - 1
end

"""
    function mpxv_r(prob_per_contact,ratio,infectivity)

Calculate `r` via route finding `p (⟨X²⟩ / ⟨X⟩) M(r) = 1`.
"""
function mpxv_r(prob_per_contact,ratio,infectivity)
    r = find_zero(r -> diff_r(r,prob_per_contact,ratio,infectivity), 0.0)
end

## Plots

plt_inf = plot(1:length(Infectivity_Profile),Infectivity_Profile,lab = "",
                xlabel = "Days after infection",
                ylabel = "Relative infectivity",
                title = "Infectiousness profile")



plt_growth_rate = plot(p-> mpxv_r(p,ratio,infectivity),0.7,1, lab = "",
        xlabel = "Probability of infection per sexual contact",
        ylabel = "Exponential growth rate (per day)",
        title = "Exp. growth rate (MSM contacts only)")

   
## Idealised case Distributions

p_choose_size_bias = normalize(p_choose .* natsal_data.snonewp,1)
random_sample_ages = natsal_data.dage[rand(Categorical(p_choose_size_bias),1_000_000)]
random_sample_num_sex = natsal_data.snonewp[rand(Categorical(p_choose_size_bias),1_000_000)]
random_unbiasedsample_ages = natsal_data.dage[rand(Categorical(p_choose),1_000_000)]

histogram(random_sample_ages,norm = :pdf,bins = 10,
            title = "Idealized age distribution of cases",
            xlabel = "Age (years)",
            ylabel = "Density in sample",
            lab= "Contact biased",
            fillalpha = 0.3)
histogram!(random_unbiasedsample_ages,norm = :pdf,bins = 10,fillalpha = 0.3, lab = "Contact unbiased" )

# histogram(random_sample_num_sex,norm = :pdf)

