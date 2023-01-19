
## UK group sizes

N_uk = 67.22e6
prop_ovr18 = 0.787
prop_men = 0.5
prop_msm = 0.034 #https://www.ons.gov.uk/peoplepopulationandcommunity/culturalidentity/sexuality/bulletins/sexualidentityuk/2020
prop_sexual_active = 1 - 0.154 #A dynamic power-law sexual network model of gonorrhoea outbreaks, Whittles et al

N_msm = round(Int64, N_uk * prop_men * prop_ovr18 * prop_msm * prop_sexual_active) #~760k

## Incubation period

# Using  this  best-fitting  distribution,  the  mean  incubation period was estimated 
# to be 8.5 days (95% credible intervals (CrI): 6.6–10.9 days), 
# with the 5th percentile of 4.2 days and the 95th percentile of 17.3 days (Table 2)

# d_incubation = Gamma(6.77, 1 / 0.77)#Fit from rerunning - older version
d_incubation = Weibull(1.4, 8.5)
negbin_std = fill(0.0, 7)
for r = 1:7
    p = r / mean(d_incubation)
    negbin_std[r] = std(NegativeBinomial(r, p))
end

plt_incfit = bar(
    negbin_std,
    title = "Discrete time model vs data-driven model for incubation",
    lab = "",
    xticks = 1:7,
    xlabel = "Number of stages",
    ylabel = "Std. incubation (days)",
    size = (800, 600),
    dpi = 250,
    legend = :top,
    left_margin = 5mm,
    guidefont = 16,
    tickfont = 13,
    titlefont = 17,
    legendfont = 12,
    right_margin = 5mm,
)

hline!(plt_incfit, [std(d_incubation)], lab = "std. of data-driven model")

__, n_optimal = findmin(abs.(negbin_std .- std(d_incubation)))
#Optimal choice is 4 stages with effective rate to match the mean
p_incubation = n_optimal / mean(d_incubation)
α_incubation_eff = -log(1 - p_incubation)
prob_bstfit =
    [zeros(n_optimal - 1); [pdf(NegativeBinomial(n_optimal, p_incubation), k) for k = 0:48]]

bar!(
    plt_incfit,
    prob_bstfit,
    lab = "",
    color = :green,
    xlabel = "Incubation period (days)",
    ylabel = "Probability",
    inset = bbox(0.65, 0.25, 0.275, 0.3, :top, :left),
    subplot = 2,
    grid = nothing,
    title = "",
    yguidefont = 15,
)

# Discretised pdf

daily_incubation_prob =
    [cdf(d_incubation, t) - cdf(d_incubation, t - 1) for t = 1:length(prob_bstfit)]

# plot!(
#     plt_incfit,
#     daily_incubation_prob,
#     lab="",
#     subplot = 2,
#     lw = 3
# )


display(plt_incfit)
savefig(plt_incfit, "plots/incubation_fit_revised.png")
## Next generation calculations

function next_state_mat(p_inf)
    [
        (1-p_incubation) p_incubation 0 0
        0 (1-p_incubation) p_incubation 0
        0 0 (1-p_inf) p_inf
        0 0 0 1
    ]
end

function calculate_next_gen_distrib(ϵ, p_inf; n_max = 50)
    T = next_state_mat(p_inf)
    state = [1.0 0 0 0]
    inf_vect = [0.0, ϵ, 1.0, 0.0]
    # inf_vect = [1.0, 1.0, 0.0, 0.0]
    p_nextgen = zeros(n_max)
    for n = 1:n_max
        p_nextgen[n] = sum(state * inf_vect)
        state = state * T
    end
    normalize!(p_nextgen, 1)
    mean_ng = sum(collect(1:n_max) .* p_nextgen)
    std_ng = sqrt(sum(collect(1:n_max) .^ 2 .* p_nextgen) - mean_ng^2)
    return mean_ng, std_ng, p_nextgen
end

##

using Roots
prop_presymptomatic = 0.5
epsilon = find_zero(
    ϵ ->
        calculate_next_gen_distrib(
            ϵ,
            (p_incubation / ϵ) * (prop_presymptomatic / (1 - prop_presymptomatic)),
        )[1] - 9.25,
    (0.2, 1.0),
)

p_inf = (p_incubation / epsilon) * (prop_presymptomatic / (1 - prop_presymptomatic))
γ_eff = -log(1 - p_inf)
mean_ng, std_ng, p_nextgen = calculate_next_gen_distrib(
    epsilon,
    (p_incubation / epsilon) * (prop_presymptomatic / (1 - prop_presymptomatic)),
)
d_serial_interval = Gamma(0.7886, 1 / 0.0853)

shape_scaler = find_zero(a -> std(Gamma(a * 0.7886, 1 / (a * 0.0853))) - std_ng, 1.0)

p_serial_interval = [cdf(d_serial_interval, t) - cdf(d_serial_interval, t - 1) for t = 1:50]
p_serial_interval_disp = [
    cdf(Gamma(shape_scaler * 0.7886, 1 / (shape_scaler * 0.0853)), t) -
    cdf(Gamma(shape_scaler * 0.7886, 1 / (shape_scaler * 0.0853)), t - 1) for t = 1:50
]
##

plt_gen_distrib = scatter(
    p_nextgen,
    lab = "Generation distribution (model)",
    ylabel = "Probability",
    xlabel = "Days after infection",
    alpha = 0.6,
    ms = 6,
    title = "Next generation distribution",
    size = (800, 600),
    dpi = 250,
    left_margin = 5mm,
    guidefont = 16,
    tickfont = 13,
    titlefont = 24,
    legendfont = 12,
    right_margin = 5mm,
)

scatter!(
    plt_gen_distrib,
    p_serial_interval,
    lab = "Serial interval (Ward et al 2022)",
    ms = 6,
    alpha = 0.6,
)
scatter!(
    plt_gen_distrib,
    p_serial_interval_disp,
    ms = 6,
    lab = "Serial interval (fixed mean, model std)",
    alpha = 0.6,
)
display(plt_gen_distrib)
savefig(plt_gen_distrib, "plots/generation_distribution.png")

## Set up equipotential groups

n_grp = 10
α_scaling = 0.81
x_min = 1.0
x_max = 3650.0

#Mean rate of yearly contacts over all MSMS
X̄ =
    (α_scaling / (α_scaling - 1)) * (x_min^(1 - α_scaling) - x_max^(1 - α_scaling)) /
    (x_min^(-α_scaling) - x_max^(-α_scaling))

#Calculation
C = (x_min^(-α_scaling) - x_max^(-α_scaling)) * X̄ / n_grp * ((1 - α_scaling) / α_scaling)
xs = zeros(n_grp + 1)
xs[1] = x_min
for k = 2:n_grp
    xs[k] = (xs[k-1]^(1 - α_scaling) + C)^(1 / (1 - α_scaling))
end
xs[end] = x_max

#Percentages in each group
ps =
    map(
        x ->
            (x_min^(-α_scaling) - x^(-α_scaling)) /
            (x_min^(-α_scaling) - x_max^(-α_scaling)),
        xs,
    ) |> diff

#Mean daily contact rates within groups
xs_pairs = [(xs[i], xs[i+1]) for i = 1:(length(xs)-1)]
mean_daily_cnts =
    map(
        x ->
            (α_scaling / (α_scaling - 1)) * (x[1]^(1 - α_scaling) - x[2]^(1 - α_scaling)) /
            (x[1]^(-α_scaling) - x[2]^(-α_scaling)),
        xs_pairs,
    ) .|> x -> x / 365.25

##Plot sexual contact groups
plt_ps = bar(
    ps,
    yscale = :log10,
    title = "Proportion MSM in each group",
    xticks = 1:10,
    ylabel = "Proportion",
    xlabel = "Sexual activity group",
    lab = "",
)
plt_μs = bar(
    mean_daily_cnts,
    yscale = :log10,
    title = "Mean daily contact rates in each group",
    xticks = 1:10,
    ylabel = "Rate (days)",
    xlabel = "Sexual activity group",
    lab = "",
)
hline!(plt_μs, [1 / 31], lab = "Vac. threshold", lw = 3, legend = :topleft)
plt = plot(plt_ps, plt_μs, size = (1000, 400), bottom_margin = 5mm, left_margin = 5mm)
display(plt)
# savefig(plt, "plots/sexual_activity_groups.png")


## Set up constant data for ABC

ingroup = 0.99
n_cliques = 50
ts = wks .|> d -> d - Date(2021, 12, 31) .|> t -> t.value

wkly_vaccinations = [
    [zeros(12); 1000; 2000; fill(5000, 4)] * 1.675
    fill(650, 20)
]

constants = [
    N_uk,
    N_msm,
    ps,
    mean_daily_cnts,
    ingroup,
    ts,
    α_incubation_eff,
    γ_eff,
    epsilon,
    n_cliques,
    wkly_vaccinations[3:end], #This because starting on week 3 
    (0.85 + 0.7) / 2,
    204,
    2,
] #Constant values passed to the MPX model

constants_no_vaccines = [
    N_uk,
    N_msm,
    ps,
    mean_daily_cnts,
    ingroup,
    ts,
    α_incubation_eff,
    γ_eff,
    epsilon,
    n_cliques,
    zeros(1000), #This because starting on week 3 
    (0.85 + 0.7) / 2,
    204,
    2,
] #Constant values passed to the MPX model - with no vaccines

## Output constant parameter values to latex definitions
const_param_tex = raw"\newcommand{\constparamtable}{" *
raw"""
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}clll@{\extracolsep{\fill}}}
\hline
Parameter   & \multicolumn{1}{c}{Fixed value}  & \multicolumn{1}{c}{Description} \\ 
\hline
\begin{tabular}[c]{@{}l@{}} Mean \\ generation time.\end{tabular} & """ * 
string(mean(d_serial_interval) |> x -> round(x, sigdigits = 3)) * " days" *
raw""" & \begin{tabular}[c]{@{}l@{}}Mean time to \\ secondary infections \cite{ward2022transmission}.\end{tabular} \\ 
\hline
\begin{tabular}[c]{@{}l@{}} Mean \\ incubation time.\end{tabular} & """ * 
string(mean(d_incubation) |> x -> round(x, sigdigits = 3)) * " days" *
raw""" & \begin{tabular}[c]{@{}l@{}}Mean time from \\ infection to symptoms \cite{ward2022transmission}.\end{tabular} \\ 
\hline
$p_{inc}$ &""" * string(p_incubation |> x -> round(x, sigdigits=3)) * 
raw""" & \begin{tabular}[c]{@{}l@{}} Daily probability \\ of latency progression (fitted).\end{tabular} \\ 
\hline
$\epsilon & """ * string(round(epsilon, sigdigits = 3)) *
raw""" & \begin{tabular}[c]{@{}l@{}} Relative infectiousness of\\ of pre-symptomatic infected \\ (fitted to 50\% pre-symptomatic infections \\ with fixed mean generation time).\end{tabular} \\
\hline
$p_{inf}$ & """ * string(p_inf |> x -> round(x, sigdigits=3)) *
raw""" & \begin{tabular}[c]{@{}l@{}} Daily probability of \\ progression from symptomatic infectious (fitted).\end{tabular} \\
\hline
\begin{tabular}[c]{@{}l@{}} Effective \\ infectious period.\end{tabular} & """ * 
((epsilon / p_incubation) + (1 / p_inf) |> x -> round(x, sigdigits = 3) |> string) * " days" *
raw""" & \begin{tabular}[c]{@{}l@{}} Effective infectious duration.\end{tabular} \\
\hline
$N$ & """ * string(round(N_uk / 1e6, sigdigits = 3)) * " millions" *
raw"""
& \begin{tabular}[c]{@{}l@{}} Population size of United Kingdom.\end{tabular} \\
\hline
$N_{gbmsm}$ & """ * string(round(N_msm / 1e3, sigdigits = 3)) * " thousands" *
raw"""
& \begin{tabular}[c]{@{}l@{}} GBMSM population size \\ of United Kingdom \cite{Whittles2019,Sexual-orientation}.\end{tabular} \\
\hline
$f(k) \sim k^{-\alpha}$ & $\alpha = $ """ * string(1 + α_scaling) *
raw"""
& \begin{tabular}[c]{@{}l@{}} Power-law distribution for rate of new \\ sexual contacts for GBMSM people \cite{Whittles2019}.\end{tabular} \\
\hline
$v_\mathrm{eff}$ & $\mathcal{U}(0.7,0.85)$ & \begin{tabular}[c]{@{}l@{}} Reduction in susceptibility of vaccinated people \\ assumed to be lower than fully vaccinated estimate \\ of 85\% \cite{Fine1988,Jezek1988}.\end{tabular} \\
\hline
$T_2$ & 23rd July & \begin{tabular}[c]{@{}l@{}} Date of WHO announcement that \\ Mpox is a public health emergency of international \\ concern (PHEIC) \cite{who-director-general-declares} \end{tabular} \\
\hline
$T_r$ & 15th Sept or 13th Oct & \begin{tabular}[c]{@{}l@{}} Mid-point times for reversion to \\ baseline behaviour (scenarios).  \end{tabular} \\
\hline""" * 
raw"""\end{tabular*}""" * "}\n"

##

open("fixed_params.tex", "w") do io
    write(io, const_param_tex)
end;
