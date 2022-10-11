using CSV, DataFrames, Plots, Dates, Plots.PlotMeasures

##
mpxv_data = CSV.File("data/mpxv_latest.csv") |> DataFrame
UK_mpxv_data = mpxv_data[[c ∈ ["England", "Wales", "Scotland", "Northern Ireland"] for c in mpxv_data.Country], :]

#Find least missing date col
sum(ismissing.(UK_mpxv_data.Date_confirmation)) # 0
sum(ismissing.(UK_mpxv_data.Date_onset)) #2429
idxs_confirmed = UK_mpxv_data.Status .== "confirmed"

##Create weekly report data
reported_msm_prop = [Date(2022,6,22) 0.96
                        Date(2022,7,6) 0.962;
                        Date(2022,7,19) 0.965;
                        Date(2022,8,1) 0.953] #https://www.gov.uk/government/publications/monkeypox-outbreak-technical-briefings/investigation-into-monkeypox-outbreak-in-england-technical-briefing-5



first_day_wk_reported = Dates.firstdayofweek.(UK_mpxv_data.Date_confirmation[idxs_confirmed])
wks = sort(unique(first_day_wk_reported))
mpxv_wkly = [sum(first_day_wk_reported .== wk) for wk in wks] .* [fill(reported_msm_prop[end,2],length(wks)) fill(1 - 0.965,length(wks))]

#Plot data
scatter(wks, mpxv_wkly, lab=["MSM" "non-MSM"],
        title="Weekly reported MPXV cases",
        ylabel="reported cases")

##Create daily report data
days = minimum(unique(UK_mpxv_data.Date_confirmation)):Day(1):maximum(unique(UK_mpxv_data.Date_confirmation))
mpxv_dly = [sum(UK_mpxv_data.Date_confirmation[idxs_confirmed] .== day) for day in days]
#Plot data
scatter(days, mpxv_dly, lab="",
        title="Daily reported MPXV cases",
        ylabel="reported cases")

## Transform data into a spread
non_zero_days = findall(mpxv_dly .> 0)
_mpxv_dly = zeros(size(mpxv_dly))
for (idx,d) in enumerate(non_zero_days)
        if idx == 1
                _mpxv_dly[1] = mpxv_dly[1]
        else
                f = (non_zero_days[idx-1]+1):d
                _mpxv_dly[f] .= mpxv_dly[d]/(d - non_zero_days[idx-1])
        end
end

scatter(days,_mpxv_dly,lab="",
        title="Daily reported MPXV cases with back filling",
        ylabel="reported cases")

##

grouped_mpxv = [sum(_mpxv_dly[Dates.firstdayofweek.(days) .== wk]) for wk in wks]
scatter(wks,grouped_mpxv,lab="",
        title="Weekly reported MPXV cases with back filling and wk grouping",
        titlefont = 11,
        ylabel="reported cases")

##
shift = 5
grouped_mpxv_fri = [sum(_mpxv_dly[Dates.firstdayofweek.(days) + Day(shift) .== wk]) for wk in (wks .+ Day(shift))]
scatter(wks,grouped_mpxv_fri,lab="",
        title="Weekly reported MPXV cases with back filling and wk grouping",
        titlefont = 11,
        ylabel="reported cases")
