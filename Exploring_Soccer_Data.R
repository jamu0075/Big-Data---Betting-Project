library(data.table)
library(dplyr)
data_path <- "/Users/nickvarberg/Downloads/beat-the-bookie-odds-series-football-dataset/"

##### change data ####
##odds_series.csv 
series      <- fread(paste0(data_path, "odds_series.csv"))
series_matches <- fread(paste0(data_path, "odds_series_matches.csv"))

## odds_series_b.csv
series_b <- fread(paste0(data_path, "odds_series_b.csv"))
series_b_matches <- fread(paste0(data_path, "odds_series_b_matches.csv"))

## merge data
matches_all <- full_join(series_matches, series_b_matches)
series_all <- full_join(series, series_b)
matches_all <- matches_all %>% filter(match_id %in% series_all$match_id)

## premier league tables
#matches_premier <- matches_all %>% filter(league == "England: Premier League")
#series_premier  <- series_all %>% filter(match_id %in% matches_premier$match_id)
#write.csv(series_premier, file = "series_premier_league.csv")
#write.csv(matches_premier, file = "matches_premier_league.csv")
# fread to get premier league tables

#### Find best bookkeeper for premier league ####
num = 216
min = 9999
for (bookkeeper in c(1:32)){
  little <- series_premier[,(6+(bookkeeper-1)*num):(6+bookkeeper*num - 1)]
  num_NaN    <- sum(little == "NaN")
  if (num_NaN < min){
    min = num_NaN
    min_book = bookkeeper
  }
  if (bookkeeper == 32){print(min_book)}
}
# bet395 (bookkeeper 9) has the most data
bookkeeper = 9
series_premier_bet365 <- series_premier[,c(1:5,(6+(bookkeeper-1)*num):(6+bookkeeper*num - 1))]
series_premier_bet365

#### Create some features for premier league bet365####
bet365 <- fread("series_premier_league_bet365.csv")
bet365 <- bet365 %>% 
              mutate(home_opening = home_b9_71) %>%
              mutate(home_closing = home_b9_0) %>%
              mutate(draw_opening = draw_b9_71) %>%
              mutate(draw_closing = draw_b9_0) %>%
              mutate(away_opening = away_b9_71) %>%
              mutate(away_closing = away_b9_0) %>%
              mutate(home_opening_minus_closing = home_opening - home_closing) %>%
              mutate(draw_opening_minus_closing = draw_opening - draw_closing) %>%
              mutate(away_opening_minus_closing = away_opening - away_closing) 

# reorder columns
#df %>% select(var, everything())