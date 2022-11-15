
# After backfilling aggregate by week starting from Mondays
cases$date_reported_to_imt_days = date2int(cases$date_reported_to_imt, units ='days')

days = min(cases$date_reported_to_imt_days):max(cases$date_reported_to_imt_days)

daily.curves = data.frame(
  day = days,
  backfilled.curve = rep(0, length(days)),
  curve = rep(0, length(days)),
  #
  backfilled.curve.msm = rep(0, length(days)),
  curve.msm = rep(0, length(days)),
  #
  backfilled.curve.nonmsm = rep(0, length(days)),
  curve.nonmsm = rep(0, length(days)),
  #
  backfilled.curve.namsm = rep(0, length(days)),
  curve.namsm = rep(0, length(days))
)

tab = sort(unique(cases$date_reported_to_imt_days))
for(i in 1:length(tab)){
  day = tab[i]
  idx_all = cases$date_reported_to_imt_days == day #all
  daily.curves$curve[day - 38] = sum(idx_all)

  idx_all = (cases$date_reported_to_imt_days == day) & (as.character(cases$f_gbmsm) == 'Yes') #msm
  daily.curves$curve.msm[day - 38] = sum(idx_all)

  idx_all = (cases$date_reported_to_imt_days == day) & (as.character(cases$f_gbmsm) == 'No') #nonmsm
  daily.curves$curve.nonmsm[day - 38] = sum(idx_all)
  
  idx_all = (cases$date_reported_to_imt_days == day) & (as.character(cases$f_gbmsm) == 'No information') #namsm
  daily.curves$curve.namsm[day - 38] = sum(idx_all)
}

backfill<-function(tab, x){
  backfilled = x
  for(i in 2:length(tab)){
    tt = (tab[i-1]+1):tab[i] - 38
    backfilled[tt] = x[tab[i] - 38] / length(tt)
  }
  return(backfilled)
}

# Perform backfilling for total number of cases
tab = sort(unique(cases$date_reported_to_imt_days))
daily.curves$backfilled.curve = backfill(tab, daily.curves$curve)


# Perform backfilling for GBMSM
daily.curves$backfilled.curve.msm = backfill(tab, daily.curves$curve.msm)


# Perform backfilling for nonGBMSM
daily.curves$backfilled.curve.nonmsm = backfill(tab, daily.curves$curve.nonmsm)


# Perform backfilling for naGBMSM
daily.curves$backfilled.curve.namsm = backfill(tab, daily.curves$curve.namsm)

weeks = min(cases$date_reported_to_imt_weeks):max(cases$date_reported_to_imt_weeks)
weekly.curves = data.frame(
  week = weeks,
  curve = rep(0, length(weeks)),
  curve.msm = rep(0, length(weeks)),
  curve.namsm = rep(0, length(weeks)),
  curve.nonmsm = rep(0, length(weeks))  
)

for(i in 1:nrow(weekly.curves)){
  week = weekly.curves$week[i]
  idx = (7*i):(7*i + 7) - 6 - 4
  idx = idx[idx>0]

  weekly.curves$curve[i] = sum(daily.curves$backfilled.curve[idx], na.rm=T)
  weekly.curves$curve.msm[i] = sum(daily.curves$backfilled.curve.msm[idx], na.rm=T)
  weekly.curves$curve.nonmsm[i] = sum(daily.curves$backfilled.curve.nonmsm[idx], na.rm=T)
  weekly.curves$curve.namsm[i] = sum(daily.curves$backfilled.curve.namsm[idx], na.rm=T)  
}

png('epidemic_curve.png')
plot(daily.curves$day, daily.curves$curve, type='l', xlab='Day', ylab='No. cases')
lines(daily.curves$day, daily.curves$backfilled.curve, col='red', lwd=1)
points(weekly.curves$week * 7 + 7, weekly.curves$curve / 7, col='red', cex=1.1, pch=19 )
dev.off()
# 
# 
png('epidemic_curve_msm.png')
plot(daily.curves$day, daily.curves$curve.msm, type='l', xlab='Day', ylab='No. cases')
lines(daily.curves$day, daily.curves$backfilled.curve.msm, col='red', lwd=1)
points(weekly.curves$week * 7 + 7, weekly.curves$curve.msm / 7, col='red', cex=1.1, pch=19 )
dev.off()
# 
# 
# 
png('epidemic_curve_nonmsm.png')
plot(daily.curves$day, daily.curves$curve.nonmsm, type='l', xlab='Day', ylab='No. cases')
lines(daily.curves$day, daily.curves$backfilled.curve.nonmsm, col='red', lwd=1)
points(weekly.curves$week * 7 + 7, weekly.curves$curve.nonmsm / 7, col='red', cex=1.1, pch=19 )
dev.off()
# 
# 
png('epidemic_curve_namsm.png')
plot(daily.curves$day, daily.curves$curve.namsm, type='l', xlab='Day', ylab='No. cases')
lines(daily.curves$day, daily.curves$backfilled.curve.namsm, col='red', lwd=1)
points(weekly.curves$week * 7 + 7, weekly.curves$curve.namsm / 7, col='red', cex=1.1, pch=19 )
dev.off()


weekly.curves$index = weekly.curves$week
weekly.curves$week = int2date(weekly.curves$index, units = 'weeks')
names(weekly.curves) = c('week', 'curve', 'gbmsm', 'na_gbmsm', 'nongbmsm', 'index')

sequential.imputation$week = NULL

# Merge with sequential imputation
weekly_data_imputation = merge(weekly.curves, sequential.imputation, by='index', all = T)

write.table(weekly_data_imputation,
            file = sprintf('X:/analysis/csv/weekly_data_imputation_%s.csv', as.character(Sys.Date())),
            sep=',',
            row.names = F)



