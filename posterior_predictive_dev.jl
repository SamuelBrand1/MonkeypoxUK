
# chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value #Announcement of Public health emergency
# inf_duration_red = 0.0

# param_draws_no_behav = param_draws .|> θ -> [θ[1:(end-4)]; zeros(4)]

# interventions_ensemble = [
#     (
#         trans_red2=θ[9] * θ[11],
#         vac_effectiveness=rand(Uniform(0.7, 0.85)),
#         trans_red_other2=θ[10] * θ[12],
#         wkly_vaccinations,
#         chp_t2,
#         inf_duration_red,
#     ) for θ in param_draws
# ]


# no_vac_ensemble = [
#     (
#         trans_red2=θ[9] * θ[11],#Based on posterior for first change point with extra dispersion
#         vac_effectiveness=rand(Uniform(0.7, 0.85)),
#         trans_red_other2=θ[10] * θ[12],
#         wkly_vaccinations=zeros(size(wkly_vaccinations_ceased)),
#         chp_t2,
#         inf_duration_red,
#     ) for θ in param_draws
# ]

# no_vac_and_no_red_ensemble = [
#     (
#         trans_red2=0,#Based on posterior for first change point with extra dispersion
#         vac_effectiveness=rand(Uniform(0.7, 0.85)),
#         trans_red_other2=θ[10] * θ[12],
#         wkly_vaccinations=zeros(size(wkly_vaccinations_ceased)),
#         chp_t2,
#         inf_duration_red,
#     ) for θ in param_draws_no_behav
# ]

# mpx_sim_function_interventions = MonkeypoxUK.mpx_sim_function_interventions

# preds_and_incidence_interventions = map(
#     (θ, intervention) ->
#         mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
#     param_draws,
#     interventions_ensemble,
# )
# preds_and_incidence_interventions_4wkrev = map(
#     (θ, intervention) ->
#         mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 4)[2:4],
#     param_draws,
#     interventions_ensemble,
# )
# preds_and_incidence_interventions_12wkrev = map(
#     (θ, intervention) ->
#         mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 12)[2:4],
#     param_draws,
#     interventions_ensemble,
# )

# preds_and_incidence_interventions_cvac = map(
#     (θ, intervention) ->
#         mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
#     param_draws,
#     no_vac_ensemble,
# )
# preds_and_incidence_interventions_cvac_4wkrev = map(
#     (θ, intervention) ->
#         mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 4)[2:4],
#     param_draws,
#     no_vac_ensemble,
# )
# preds_and_incidence_interventions_cvac_12wkrev = map(
#     (θ, intervention) ->
#         mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 12)[2:4],
#     param_draws,
#     no_vac_ensemble,
# )

# preds_and_incidence_novac_no_chg = map(
#     (θ, intervention) ->
#         mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
#     param_draws_no_behav,
#     no_vac_and_no_red_ensemble,
# )


##Gather projections
d1, d2 = size(mpxv_wkly)

preds = [particle.other.detected_cases for particle in smc.particles]
# preds_4wk = [x[1] for x in preds_and_incidence_interventions_4wkrev]
# preds_12wk = [x[1] for x in preds_and_incidence_interventions_12wkrev]

# preds_cvac = [x[1] for x in preds_and_incidence_interventions_cvac]
# preds_cvac4wk = [x[1] for x in preds_and_incidence_interventions_cvac_4wkrev]
# preds_cvac12wk = [x[1] for x in preds_and_incidence_interventions_cvac_12wkrev]

# pred_unmitigated = [x[1] for x in preds_and_incidence_novac_no_chg]
# cum_cases_unmitigated = [cumsum(x[1], dims=1) for x in preds_and_incidence_novac_no_chg]

# cum_cases_forwards =
#     [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions]
# cum_cases_forwards_4wk = [
#     cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions_4wkrev
# ]
# cum_cases_forwards_12wk = [
#     cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions_12wkrev
# ]

# cum_cases_forwards_cvac =
#     [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions_cvac]
# cum_cases_forwards_cvac4wk = [
#     cumsum(x[1][(d1+1):end, :], dims=1) for
#     x in preds_and_incidence_interventions_cvac_4wkrev
# ]
# cum_cases_forwards_cvac12wk = [
#     cumsum(x[1][(d1+1):end, :], dims=1) for
#     x in preds_and_incidence_interventions_cvac_12wkrev
# ]

##Simulation projections

cred_int = MonkeypoxUK.cred_intervals(preds)
# cred_int_4wk = MonkeypoxUK.cred_intervals(preds_4wk)
# cred_int_12wk = MonkeypoxUK.cred_intervals(preds_12wk)

# cred_int_cvac = MonkeypoxUK.cred_intervals(preds_cvac)
# cred_int_cvac4wk = MonkeypoxUK.cred_intervals(preds_cvac4wk)
# cred_int_cvac12wk = MonkeypoxUK.cred_intervals(preds_cvac12wk)

# cred_int_unmitigated = MonkeypoxUK.cred_intervals(pred_unmitigated)
# cred_int_cum_cases_unmitigated = MonkeypoxUK.cred_intervals(cum_cases_unmitigated)
