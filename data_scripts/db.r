library(RODBC)

run_query<-function(query, server = "SQLCLUSCOLHPO19\\HPO19"){
  conn = odbcDriverConnect(paste0("Driver=SQL Server; Server=",server,";"))
  df = sqlQuery(conn, query)
  writeLines(paste0("There are ",nrow(df)," rows and ", ncol(df)," columns"))
  # close the connection
  odbcClose(conn)
  return(df)
}
# query = "SELECT * FROM [S303_Monkeypox_modelling_PID].[dbo].[M2_manual_case_linelist]"
# df = run_query(query)
