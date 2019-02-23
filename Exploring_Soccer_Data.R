library(data.table)
data_path <- "/Users/nickvarberg/Downloads/beat-the-bookie-odds-series-football-dataset/"

#### closing_odds.csv ####
closing_name <- "closing_odds.csv"
closing      <- fread(paste0(data_path, closing_name))

# Columns
colnames(closing)
head(closing)
range(closing$match_date)
table(closing$n_odds_home_win)
table(closing$n_odds_away_win)
table(closing$league)

##### odds_series.csv ####
series_name <- "odds_series.csv"
series      <- fread(paste0(data_path, series_name))
colnames(series)[10:20]
series[1:2,1:10]
range(series$match_date)
series[1:2,]