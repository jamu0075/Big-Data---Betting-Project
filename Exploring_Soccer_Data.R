library(data.table)
library(dplyr)
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
range(closing$match_id)

##### odds_series.csv ####
series_name <- "odds_series.csv"
series      <- fread(paste0(data_path, series_name))
series_matches <- fread(paste0(data_path, "odds_series_matches.csv"))
colnames(series)[10:20]
series[1:2,1:10]
range(series$match_date)
series[1:2,(ncol(series)-10):ncol(series)]
tbl <- table(series$league)
order(tbl)
world_club_friendly <- series %>% filter(league == "World: Club Friendly")

##### odds_series_b.csv #####
series_b_name <- "odds_series_b.csv"
series_b <- fread(paste0(data_path, series_b_name))
colnames(series_b)[1:10]
range(series_b$match_id)
range(series_b$match_date)
series_b_matches <- fread(paste0(data_path, "odds_series_b_matches.csv"))
range(series_b_matches$match_id)
colnames(series_b_matches)
table(series_b_matches$league)
table(series_matches$league)

##### merged data ####
matches_all <- full_join(series_matches, series_b_matches)
series_all <- full_join(series, series_b)
matches_all <- matches_all %>% filter(match_id %in% series_all$match_id)
matches_premier <- matches_all %>% filter(league == "England: Premier League")
series_premier  <- series_all %>% filter(match_id %in% matches_premier$match_id)