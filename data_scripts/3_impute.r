# optimal.params = matrix(data=c(
#     8, 0.12, 0.03, 0.7, 0.5, 10, 0.7, 23, 0,
#    12, 0.17, 0.00, 0.7, 0.5, 5, 0.7, 23, 0,
#    16, 0.16, 0.01, 0.5, 0.7, 10, 0.5, 18, 0,
#    20, 0.15, 0.00, 0.5, 0.4, 10, 0.3, 28, 0,
#    24, 0.15, 0.00, 0.5, 0.7, 4, 0.3, 23, 0,
#    26, 0.15, 0.00, 0.5, 0.7, 9, 0.5, 18, 0),
#    ncol = 9, byrow = T)

optimal.params = matrix(data=c(
  28, 0.16, 0.01, 0.5, 0.6, 7, 0.3, 18, 0,
  29, 0.16, 0.01, 0.7, 0.7, 5, 0.5, 23, 0),
  ncol = 9, byrow = T)

optimal.params = data.frame(optimal.params)
names(optimal.params) = c("tt","scores1", "scores2",
                      "subsample","colsample_bytree","max_depth","eta","nrounds","min_child_weight")

to_impute = as.character(cases$f_gbmsm) == "No information"
to_learn = !to_impute

# Only keep fields that do not contain too many NAs:
n.na = apply(df, 2, function(y){sum(is.na(y))})
cols = n.na < 500

x = as.matrix(df[to_learn, cols])
y = as.integer(as.character(cases$f_gbmsm[to_learn]) == 'Yes')

x_to_impute = as.matrix(df[to_impute, cols])

Imputations = list()
RMSEs = list()

# Note: remember to remove the column which contains report week
columns_to_remove = which(colnames(x) == 'date_reported_to_imt_weeks')



for (i in 1:nrow(optimal.params)){
  tt = optimal.params$tt[i]
  idx = x[,'date_reported_to_imt_weeks'] <= tt
  #
  print(tt)
  scale_pos_weight = sqrt((length(y[idx]) - sum(y[idx])) / sum(y[idx]))
  
  data_for_learning = xgb.DMatrix(data = x[idx, -columns_to_remove], label = y[idx])
    
  nrounds = optimal.params$nrounds[i]
  max_depth = optimal.params$max_depth[i]
  eta = optimal.params$eta[i]
  subsample = optimal.params$subsample[i]
  colsample_bytree = optimal.params$colsample_bytree[i]
  min_child_weight = optimal.params$min_child_weight[i]

  watchList = list(vanilla.set = data_for_learning)
  
  idx = x_to_impute[,'date_reported_to_imt_weeks'] <= tt
  n_to_impute = sum(idx)
  Imputation = matrix(data=1:n_to_impute, nrow = n_to_impute)
  Rmse = c()
  for(ii in 1:20){
    model = xgb.train(data = data_for_learning,
                      objective='binary:logistic',
                      nrounds = nrounds,
                      verbose = 0,
                      max_depth = max_depth,
                      eta = eta,
                      subsample = subsample,
                      colsample_bytree = colsample_bytree,
                      min_child_weight = min_child_weight,
                      # fixed hyperparameters:
                      scale_pos_weight = scale_pos_weight,
                      watchlist = watchList,
                      early_stopping_rounds = 10,
                      alpha = 0.5)
    
    # predict GBMSM of data to impute.
    imputation = predict(model, xgb.DMatrix(data=x_to_impute[idx, -columns_to_remove]))
    Imputation = cbind(Imputation, imputation)
    Rmse = c(Rmse, sqrt(mean((predict(model,  data_for_learning) - getinfo(data_for_learning,'label'))^2)))
  
  }
  Imputations[[as.character(tt)]] = Imputation
  RMSEs[[as.character(tt)]] = Rmse
}

#### include last days:
# 
#   scale_pos_weight = sqrt((length(y) - sum(y)) / sum(y))
#   data_for_learning = xgb.DMatrix(data = x, label = y)
#   i=nrow(optimal.params)
#   nrounds = optimal.params$nrounds[i]
#   max_depth = optimal.params$max_depth[i]
#   eta = optimal.params$eta[i]
#   subsample = optimal.params$subsample[i]
#   colsample_bytree = optimal.params$colsample_bytree[i]
#   min_child_weight = optimal.params$min_child_weight[i]
#   
#   # watchList = list(oversampled.set=data_for_learning.,
#   #                  vanilla.set = data_for_learning)
#   
#   watchList = list(vanilla.set = data_for_learning)
#   
#   n_to_impute = nrow(x_to_impute)
#   Imputation = matrix(data=1:n_to_impute, nrow = n_to_impute)
#   Rmse = c()
#   for(i in 1:20){
#     model = xgb.train(data = data_for_learning,
#                       objective='binary:logistic',
#                       nrounds = nrounds,
#                       verbose = 0,
#                       max_depth = max_depth,
#                       eta = eta,
#                       subsample = subsample,
#                       colsample_bytree = colsample_bytree,
#                       min_child_weight = min_child_weight,
#                       # fixed hyperparameters:
#                       scale_pos_weight = scale_pos_weight,
#                       watchlist = watchList,
#                       early_stopping_rounds = 10,
#                       alpha = 0.5)
#     
#     # predict GBMSM of data to impute.
#     imputation = predict(model, xgb.DMatrix(data=x_to_impute))
#     Imputation = cbind(Imputation, imputation)
#     Rmse = c(Rmse, sqrt(mean((predict(model,  data_for_learning) - getinfo(data_for_learning,'label'))^2)))
#     
#   }
#   Imputations[[as.character(max(df[,'date_reported_to_imt'] ))]] = Imputation
#   RMSEs[[as.character(max(df[,'date_reported_to_imt'])) ]] = Rmse
#   
######

# average_imputation = list()
# for(tt in names(Imputations)){
#   
#   #take the imputations with lowest RMSE
#   best_columns = head(order(RMSEs[[tt]]), 1)
#   points(RMSEs[[tt]][best_columns])
#   
#   average_imputation[[tt]] = rowMeans( Imputations[[tt]][,best_columns+1])
#   tmp = attributes(average_imputation[[tt]])
#   tmp[['std.err']] = apply(Imputations[[tt]][,best_columns+1], 1, sd)
#   tmp[['median']] = apply(Imputations[[tt]][,best_columns+1], 1, median)
#   attributes(average_imputation[[tt]]) = tmp
# }
#######
best_imputation = list()
for(tt in names(Imputations)){

  # Keep optimal imputation
  best_column = which.min(RMSEs[[tt]])
  best_imputation[[tt]] = Imputations[[tt]][,best_column+1]
}


# Aggregate by week starting from Mondays.
average_imputation_week = list()
for(tt in names(best_imputation)){
  
  week.range = c(min(df[to_impute, "date_reported_to_imt_weeks"]), as.numeric(tt))
  
  idx = x_to_impute[,'date_reported_to_imt_weeks'] <= as.numeric(tt)
  df_to_impute = x_to_impute[idx,]
  
  P.w = c()
  P.w.lower = c()
  P.w.upper = c()
  for (week in week.range[1]:week.range[2]){
    idx = df_to_impute[,'date_reported_to_imt_weeks'] == week
    

    tmp = vector(mode = 'numeric', length=100000)
    tmp2 = vector(mode = 'numeric', length=100000)
    len = length(best_imputation[[tt]][idx])
    for(i in 1:100000){
      tmp[i] = sum(rbinom(len, 1, best_imputation[[tt]][idx]))
    }
    
    P.w = c(P.w, mean(tmp) / len)
    CIs = quantile(tmp, c(0.025, 0.975))
    print(c(mean(best_imputation[[tt]][idx]), mean(tmp), len, CIs))
    #
    P.w.lower = c(P.w.lower, CIs[1] / len)
    P.w.upper = c(P.w.upper, CIs[2] / len)
  }
  average_imputation_week[[tt]] = data.frame(week = week.range[1]:week.range[2],
                                             P.w = P.w,
                                             P.w.lower = P.w.lower,
                                             P.w.upper = P.w.upper
                                             )
}


int2date <- function(w, units = 'weeks'){
  as.character(first_date + as.difftime(w, units = units))
}
Vint2date <- Vectorize(int2date)


# Create data.frame to export
last_week = names(average_imputation_week)[length(names(average_imputation_week))]
last_week_int = as.integer(last_week)
sequential.imputation = data.frame(
  index = average_imputation_week[[last_week]]$week,
  week = Vint2date(average_imputation_week[[last_week]]$week, units='weeks'))

for (week in names(average_imputation_week)){
  colname = week
  week_int = as.integer(week)
  
  if (week_int == last_week_int){
    sequential.imputation[,colname] = average_imputation_week[[week]]$P.w
    colname = paste('lower', week, sep='.')
    sequential.imputation[,colname] = average_imputation_week[[week]]$P.w.lower
    colname = paste('upper', week, sep='.')
    sequential.imputation[,colname] = average_imputation_week[[week]]$P.w.upper 
  }else{
    sequential.imputation[,colname] = c(average_imputation_week[[week]]$P.w, rep(NA, last_week_int-week_int))
    colname = paste('lower', week, sep='.')
    sequential.imputation[,colname] = c(average_imputation_week[[week]]$P.w.lower, rep(NA, last_week_int-week_int))
    colname = paste('upper', week, sep='.')
    sequential.imputation[,colname] = c(average_imputation_week[[week]]$P.w.upper, rep(NA, last_week_int-week_int))
  }
}


sequential.imputation = rbind(
  sequential.imputation[1,],
  sequential.imputation
)

# At week 7 we have one backfilled naGBMSM: backfill row 8 to row 7.
sequential.imputation[1,1] = 7
sequential.imputation$week = as.character(sequential.imputation$week)
sequential.imputation[1,2] = int2date(7, units = 'weeks')

 
write.table(sequential.imputation,
            file = sprintf('X:/analysis/csv/0.95_sequential_%s.csv', as.character(Sys.Date())), sep=',', row.names = F)
 
save(best_imputation, file = paste0("best_imputation_list", as.character(Sys.Date()), ".RData"))
