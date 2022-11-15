
rm(list=ls())
library(magrittr)

source("db.r")
query1 <- "SELECT * FROM [S303_Monkeypox_modelling_PID].[dbo].[M4_mpx_case_linelist]"
cases <- run_query(query1)

# Convert dates to integer:
first_date =  as.POSIXct("28/03/2022", format="%d/%m/%Y", tz = "UTC")

date2int<-function(x, units = "weeks"){
  ret = as.POSIXct(as.character(x),format="%Y-%m-%d", tz = "UTC")
  ret = as.integer(difftime(ret,
                           first_date,
                           units = units))
  return(ret)
}

# save week of reporting date
cases$date_reported_to_imt_weeks = date2int(cases$date_reported_to_imt, units ='weeks')

# convert dates to day number
columns = c("f_symptom_onset", "q_date_symp_onset", "symptom_onset", "result_date", "lab_report_date", "date_reported_to_hpzone",
            "date_reported_to_imt","date_discarded", "date_added_to_linelist","spec_date",
            "reception_date", "return_travel_date1", "return_travel_date2", "outward_travel_date1", "outward_travel_date2")
df1 = apply(cases[,columns], 2, date2int, units = 'days' )

# perform one-hot encording of categorical features
# incident                                                                                        3
# sex                                                                                                      M
# hpt                                                                         Bedfordshire and Hertfordshire
# status                                                                                           Confirmed
# assay                                                                                            MONKEYPOX
# result_status                                                                                     POSITIVE
# gbmsm_from_ripl_or_hpzone                                                                             <NA>
# f_gbmsm                                                                                     No information
# q_ethnicity                                                                                           <NA>
# q_gender                                                                                              <NA>
# q_gender_diff_to_sex_at_birth                                                                         <NA>
# q_sex_behaviour_21_days_prior_symp_onset_1_new_partners                                               <NA>
# q_sex_behaviour_21_days_prior_symp_onset_2_oneoff_partners                                            <NA>
# q_sex_behaviour_21_days_prior_symp_onset_3_occassional_partners                                       <NA>
# q_sex_behaviour_21_days_prior_symp_onset_4_established_partners                                       <NA>
# q_sex_behaviour_21_days_prior_symp_onset_5_group_sex                                                  <NA>
# q_sex_behaviour_21_days_prior_symp_onset_6_not_uk_residents                                           <NA>
# q_sex_behaviour_21_days_prior_symp_onset_7_chemsex                                                    <NA>
# q_sex_behaviour_21_days_prior_symp_onset_8_in_locations_not_your_town                                 <NA>
# q_sex_behaviour_21_days_prior_symp_onset_10_grindr_scruff_other_apps                                  <NA>
# q_sex_behaviour_21_days_prior_symp_onset_11_cruising_grounds                                          <NA>
# q_sex_behaviour_21_days_prior_symp_onset_12_private_sex_party                                         <NA>
# q_sex_men                                                                                             <NA>
# q_gbmsm                                                                                               <NA>
# recent_foreign_travel                                                                                   No
# country1                                                                                         <NA>
# outward_travel_date1                                                                                  <NA>
# return_travel_date1                                                                                   <NA>
# country2                                                                                              <NA>
# outward_travel_date2                                                                                  <NA>
# return_travel_date2                                                                                   <NA>
# country3                                                                                              <NA>
# outward_travel_date3                                                                                  <NA>
# return_travel_date3                                                                                   <NA>
# country                                                                                            England
# utla21nm                                                                                          Thurrock
# phec15nm                                                                                   East of England
# utla21nm_hospital                                                                   Hammersmith and Fulham
# phec15nm_hospital                                                                                   London
# phec15nm_res_hosp                                                                          East of England

cases[, c("country1","country2","country","phec15nm","phec15nm_hospital","phec15nm_res_hosp", "recent_foreign_travel")] = as.data.frame(apply(
  cases[, c("country1","country2","country","phec15nm","phec15nm_hospital","phec15nm_res_hosp","recent_foreign_travel")],
  2, function(x){as.factor(tolower(as.character(x)))} ))

cases$incident = as.factor(cases$incident)

columns = c("incident", "sex", "hpt", "status", "assay",
            "result_status", "gbmsm_from_ripl_or_hpzone",
            "q_ethnicity", "q_gender", "q_gender_diff_to_sex_at_birth",
            "q_sex_behaviour_21_days_prior_symp_onset_1_new_partners",
            "q_sex_behaviour_21_days_prior_symp_onset_2_oneoff_partners",
            "q_sex_behaviour_21_days_prior_symp_onset_3_occassional_partners",
            "q_sex_behaviour_21_days_prior_symp_onset_4_established_partners",
            "q_sex_behaviour_21_days_prior_symp_onset_5_group_sex",
            "q_sex_behaviour_21_days_prior_symp_onset_6_not_uk_residents",
            "q_sex_behaviour_21_days_prior_symp_onset_7_chemsex",
            "q_sex_behaviour_21_days_prior_symp_onset_8_in_locations_not_your_town",
            "q_sex_behaviour_21_days_prior_symp_onset_10_grindr_scruff_other_apps",
            "q_sex_behaviour_21_days_prior_symp_onset_11_cruising_grounds",
            "q_sex_behaviour_21_days_prior_symp_onset_12_private_sex_party",
            "q_sex_men",
            "q_gbmsm",
            "country1",
            "country2",
            "country",
            "phec15nm",
            "phec15nm_hospital",
            "phec15nm_res_hosp",
            "date_reported_to_imt_weeks")
library(mltools)
library(data.table)
df2 = as.data.frame(one_hot(data.table(cases[,columns])))
df2$`f_gbmsm_No information`=NULL

df = cbind(df1, df2)

# We wish to impute GBMSM in f_gbmsm
writeLines("Quick check of `cases$f_gbmsm` content:")
print(unique(as.character(cases$f_gbmsm)))

to_impute = as.character(cases$f_gbmsm) == "No information"
to_learn = !to_impute

x = as.matrix(df[to_learn,])
y = as.integer(as.character(cases$f_gbmsm[to_learn]) == 'Yes')

# Remove fields that contain lot of NAs:
n.na = apply(df, 2, function(y){sum(is.na(y))})
cols = n.na < 500
x = x[,cols]

