
library(xgboost)
searchGrid = expand.grid(subsample = seq(0.5, 0.8, 0.2),
                         colsample_bytree = seq(0.4, 0.7, 0.1),
                         max_depth = seq(4, 10, 1),
                         eta = seq(0.3, 0.9, 0.2),
                         nrounds = seq(13, 30, 5),
                         min_child_weight = c(0, 3)
)

weeks_for_sequential_updates = seq(
  min(x[,'date_reported_to_imt_weeks']) + 3,
  max(x[,'date_reported_to_imt_weeks']),
  4)
if (max(x[,'date_reported_to_imt']) != weeks_for_sequential_updates[length(weeks_for_sequential_updates)]){
  weeks_for_sequential_updates = c(weeks_for_sequential_updates, max(x[,'date_reported_to_imt_weeks']))
}


optimal.params  = matrix(data = NA, nrow=0, ncol = 9)


# Perform grid search
#for (tt in weeks_for_sequential_updates){
for (tt in c(29)){
  idx = x[,'date_reported_to_imt_weeks'] <= tt
  print(tt)
  print(table(y[idx]))
  scale_pos_weight = sqrt((length(y[idx]) - sum(y[idx])) / sum(y[idx]))
  
  columns_to_remove = which(colnames(x) == 'date_reported_to_imt_weeks')
  
  data_for_learning = xgb.DMatrix(data = x[idx, -columns_to_remove], label = y[idx])
  
  scores<-t(apply(searchGrid, 1, function(parameterList){
    currentSubsampleRate <- parameterList[["subsample"]]
    currentColsampleRate <- parameterList[["colsample_bytree"]]
    currentDepth <- parameterList[["max_depth"]]
    currentEta <- parameterList[["eta"]]
    currentNumRounds <- parameterList[["nrounds"]]
    currentMinChild <- parameterList[["min_child_weight"]]
    xgboostModelCV <- xgb.cv(data =  data_for_learning, nfold = 2, verbose=F, metrics = "rmse", objective='binary:logistic',
                             nrounds = currentNumRounds,
                             max_depth = currentDepth,
                             eta = currentEta,
                             subsample = currentSubsampleRate,
                             colsample_bytree = currentColsampleRate,
                             min_child_weight = currentMinChild,
                             # fixed hyperparameters:
                             scale_pos_weight = scale_pos_weight,
                             alpha = 0.5,
                             early_stopping_rounds = 15)
    
    xvalidationScores <- tail(as.data.frame(xgboostModelCV$evaluation_log), 1)
    return(
      c(xvalidationScores$test_rmse_mean, xvalidationScores$test_rmse_std)
    )
  }
  ))
  
  png(sprintf('score_%d.png', tt))
  plot(scores[,1], scores[,2], xlab = 'RMSE', ylab = 'std')
  dev.off()
  
  #best hyperparameters are:
  i = which.min(scores[, 1])
  # scores[i,]
  # [1] 1.0000000 0.0119356
  # searchGrid[i,]
  #      subsample colsample_bytree max_depth eta nrounds min_child_weight
  # 1780       0.7              0.7         5 0.9     100                0
  cat(
    sprintf("Scores %.2f %.2f; best RMSE hyperparams at week %d are:", scores[i,1], scores[i,2], tt),
    unlist(searchGrid[i,]), '\n')
  
  optimal.params = rbind(optimal.params, c(tt, scores[i,1], scores[i,2], unlist(searchGrid[i,])))
}

