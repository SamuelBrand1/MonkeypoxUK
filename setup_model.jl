## UK group sizes

N_uk = 67.22e6
prop_ovr18 = 0.787
prop_men = 0.5
prop_msm = 0.034 #https://www.ons.gov.uk/peoplepopulationandcommunity/culturalidentity/sexuality/bulletins/sexualidentityuk/2020
prop_sexual_active = 1 - 0.154 #A dynamic power-law sexual network model of gonorrhoea outbreaks

N_msm = round(Int64, N_uk * prop_men * prop_ovr18 * prop_msm * prop_sexual_active) #~1.5m

## Incubation period

# Using  this  best-fitting  distribution,  the  mean  incuba-tion period was estimated to be 8.5 days
#  (95% credible intervals (CrI): 6.6–10.9 days), 
# with the 5th percentile of 4.2 days and the 95th percentile of 17.3 days (Table 2)

d_incubation = Gamma(6.77, 1 / 0.77)#Fit from rerunning 

negbin_std = fill(0.0, 8)
for r = 1:8
    p = r / mean(d_incubation)
    negbin_std[r] = std(NegativeBinomial(r, p))
end
plt_incfit = bar(negbin_std,
    title="Discrete time model vs data-driven model for incubation",
    lab="",
    xticks=1:8,
    xlabel="Number of stages",
    ylabel="Std. incubation (days)",
    size=(800, 600), left_margin=5mm)
hline!(plt_incfit, [std(d_incubation)], lab="std. data-driven model")
display(plt_incfit)
savefig(plt_incfit, "plots/incubation_fit.png")
#Optimal choice is 4 stages with effective rate to match the mean
p_incubation = 4 / mean(d_incubation)
α_incubation_eff = -log(1 - p_incubation)
## Set up equipotential groups

n_grp = 10
α_scaling = 0.82
x_min = 1.0
x_max = 3650.0

#Mean rate of yearly contacts over all MSMS
X̄ = (α_scaling / (α_scaling - 1)) * (x_min^(1 - α_scaling) - x_max^(1 - α_scaling)) / (x_min^(-α_scaling) - x_max^(-α_scaling))

#Calculation
C = (x_min^(-α_scaling) - x_max^(-α_scaling)) * X̄ / n_grp * ((1 - α_scaling) / α_scaling)
xs = zeros(n_grp + 1)
xs[1] = x_min
for k = 2:n_grp
    xs[k] = (xs[k-1]^(1 - α_scaling) + C)^(1 / (1 - α_scaling))
end
xs[end] = x_max

#Percentages in each group
ps = map(x -> (x_min^(-α_scaling) - x^(-α_scaling)) / (x_min^(-α_scaling) - x_max^(-α_scaling)), xs) |> diff

#Mean daily contact rates within groups
xs_pairs = [(xs[i], xs[i+1]) for i = 1:(length(xs)-1)]
mean_daily_cnts = map(x -> (α_scaling / (α_scaling - 1)) * (x[1]^(1 - α_scaling) - x[2]^(1 - α_scaling)) / (x[1]^(-α_scaling) - x[2]^(-α_scaling)), xs_pairs) .|> x -> x / 365.25

##Plot sexual contact groups
plt_ps = bar(ps,
    yscale=:log10,
    title="Proportion MSM in each group",
    xticks=1:10,
    ylabel="Proportion",
    xlabel="Sexual activity group",
    lab="")
plt_μs = bar(mean_daily_cnts,
    yscale=:log10,
    title="Mean daily contact rates in each group",
    xticks=1:10,
    ylabel="Rate (days)",
    xlabel="Sexual activity group",
    lab="")
hline!(plt_μs, [1 / 31], lab="Vac. threshold", lw=3, legend=:topleft)
plt = plot(plt_ps, plt_μs,
    size=(1000, 400),
    bottom_margin=5mm, left_margin=5mm)
display(plt)
savefig(plt, "plots/sexual_activity_groups.png")


## Set up for ABC

ingroup = 0.99
n_cliques = 50
ts = wks .|> d -> d - Date(2021, 12, 31) .|> t -> t.value
wkly_vaccinations = [zeros(12); 1000; 2000; fill(5000, 23)] * 1.5 #shifted one week to account for delay between jab and effect
constants = [N_uk, N_msm, ps, mean_daily_cnts, ingroup, ts, α_incubation_eff, n_cliques, wkly_vaccinations, 0.8, 204] #Constant values passed to the MPX model

## Check model runs

# _p = [0.01, 0.5, 20, 0.2, 0.5, 6, 1.5, 130, 0.7, 0.3, 0.5, 0.5]
# err, pred = MonkeypoxUK.mpx_sim_function_chp(_p, constants, mpxv_wkly[1:9, :])

# plt = plot(pred, color=[1 2])
# scatter!(plt, mpxv_wkly, color=[1 2])
# display(plt)
# print(err)
