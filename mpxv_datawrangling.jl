using CSV, DataFrames, Plots, Dates, Plots.PlotMeasures

##
mpxv_data = CSV.File("mpvx_latest.csv") |> DataFrame
england_mpxv_data = mpxv_data[mpxv_data.Country.=="England", :]
#Find least missing date col
sum(ismissing.(england_mpxv_data.Date_confirmation)) # 0
sum(ismissing.(england_mpxv_data.Date_onset)) #1775

##Create weekly report data
# p/(1-p) = R ⟹ p = R(1-p) ⟹ p = R/(1+R)
msm_ratio = 1.0 ./ [Inf, 0.00000000, 0.01351351, 0.02222222, 0.05769231, 0.04575163,
        0.01408451, 0.03296703, 0.03125000, 0.07575758, 0.06172840, 0.09677419]
msm_freq = msm_ratio ./ (1 .+ msm_ratio) .|> x -> isnan(x) ? 1.0 : x
msm_freq = [msm_freq; msm_freq[end,:]]
first_day_wk_reported = Dates.firstdayofweek.(england_mpxv_data.Date_confirmation)
wks = sort(unique(first_day_wk_reported))
mpxv_wkly = [sum(first_day_wk_reported .== wk) for wk in wks] .* [msm_freq 1.0 .- msm_freq]

#Plot data
scatter(wks, mpxv_wkly, lab=["MSM" "non-MSM"],
        title="Weekly reported MPXV cases",
        ylabel="reported cases")

##Create daily report data
days = minimum(unique(england_mpxv_data.Date_confirmation)):Day(1):maximum(unique(england_mpxv_data.Date_confirmation))
mpxv_dly = [sum(england_mpxv_data.Date_confirmation .== day) for day in days]
#Plot data
scatter(days, mpxv_dly, lab="",
        title="Daily reported MPXV cases",
        ylabel="reported cases")
