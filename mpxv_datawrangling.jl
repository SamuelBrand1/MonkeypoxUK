using CSV, DataFrames, Plots, Dates

##
mpxv_data = CSV.File("mpvx_latest.csv") |> DataFrame

england_mpxv_data = mpxv_data[mpxv_data.Country .== "England",:]
#Find least missing date col
sum(ismissing.(england_mpxv_data.Date_confirmation)) # 0
sum(ismissing.(england_mpxv_data.Date_onset)) #1775

##Create weekly report data
first_day_wk_reported = Dates.firstdayofweek.(england_mpxv_data.Date_confirmation)
wks = sort(unique(first_day_wk_reported))
mpxv_wkly = [ sum(first_day_wk_reported .== wk) for wk in wks]
#Plot data
scatter(wks,mpxv_wkly,lab = "",
        title = "Weekly reported MPXV cases",
        ylabel = "reported cases")

##Create daily report data
days = minimum(unique(england_mpxv_data.Date_confirmation)):Day(1):maximum(unique(england_mpxv_data.Date_confirmation))
mpxv_dly = [ sum(england_mpxv_data.Date_confirmation .== day) for day in days]
#Plot data
scatter(days,mpxv_dly,lab = "",
        title = "Daily reported MPXV cases",
        ylabel = "reported cases")
