library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
data_path <- "/Users/nickvarberg/Downloads/beat-the-bookie-odds-series-football-dataset/"

##### change data ####
##odds_series.csv 
series      <- fread(paste0(data_path, "odds_series.csv"))
series_matches <- fread(paste0(data_path, "odds_series_matches.csv"))

## odds_series_b.csv
series_b <- fread(paste0(data_path, "odds_series_b.csv"))
series_b_matches <- fread(paste0(data_path, "odds_series_b_matches.csv"))

## merge data
series_all <- full_join(series, series_b)
matches_all <- full_join(series_matches, series_b_matches)
matches_all <- matches_all %>% filter(match_id %in% series_all$match_id)
rm(series, series_b, series_matches, series_b_matches)

#### Find best bookkeeper for all leagues ####
num = 216
min = 999999999
books = c(9)
for (bookkeeper in books){
  little <- series[,(6+(bookkeeper-1)*num):(6+bookkeeper*num - 1)]
  num_NaN    <- sum(little == "NaN")
  if (num_NaN < min){
    min = num_NaN
    min_book = bookkeeper
  }
  if (bookkeeper == last(books)){ print(min)
    print(min_book)}
}
print(min/(216*31074))
# Results: min bookkeeper is 7 and has 41% NaN
#          bookkeeper 9, bet365, has 46% NaN

##### bet365 table ####
bookkeeper <- 9
bet365cols <- c(1:5,(6+(bookkeeper-1)*num):(6+bookkeeper*num - 1))
bet3651 <- series %>% select(., get("bet365cols"))
bet3652 <- series_b %>% select(., get("bet365cols"))
bet365 <- full_join(bet3651, bet3652)
rm(bet3651,bet3652)


#### Create some features for bet365####
bet365 <- fread("series_premier_league_bet365.csv")
bet365 <- bet365 %>%
  mutate(outcome = if_else(score_home >= score_away, if_else(score_home == score_away, "draw", "home"), "away")) %>%
  mutate(home_opening = home_b9_71) %>%
  mutate(home_closing = home_b9_0) %>%
  mutate(draw_opening = draw_b9_71) %>%
  mutate(draw_closing = draw_b9_0) %>%
  mutate(away_opening = away_b9_71) %>%
  mutate(away_closing = away_b9_0) %>%
  mutate(home_opening_minus_closing = home_opening - home_closing) %>%
  mutate(draw_opening_minus_closing = draw_opening - draw_closing) %>%
  mutate(away_opening_minus_closing = away_opening - away_closing) %>%
  mutate(closing_odds_outcome = if_else(outcome %in% c("home", "draw"), (if_else(outcome == "home", home_closing, draw_closing)), away_closing))

# home df
df <- bet365 %>% select(c(6:77))
min <- apply(df, 1, FUN = min)
max <- apply(df, 1, FUN = max)
range <- max - min
bet365 <- bet365 %>% 
  mutate(home_min = min) %>%
  mutate(home_max = max) %>%
  mutate(home_range = range)

# draw df
df <- bet365 %>% select(c(78:149))
min <- apply(df, 1, FUN = min)
max <- apply(df, 1, FUN = max)
range <- max - min
bet365 <- bet365 %>% 
  mutate(draw_min = min) %>%
  mutate(draw_max = max) %>%
  mutate(draw_range = range)

# away df
df <- bet365 %>% select(c(150:221))
min <- apply(df, 1, FUN = min)
max <- apply(df, 1, FUN = max)
range <- max - min
bet365 <- bet365 %>% 
  mutate(away_min = min) %>%
  mutate(away_max = max) %>%
  mutate(away_range = range)


# reorder columns
bet365_short <- bet365 %>% 
  select(c(1:3, "outcome", "closing_odds_outcome", (241-18):241))

# drop NaNs
bet365_dropped <- bet365_short %>% drop_na()

# add team ids
matches_all <- matches_all %>% select(match_id, league, home_team, away_team)
bet365_joined <- left_join(bet365_dropped, select(matches_all, match_id, match_date), by = "match_id")
bet365_joined <- bet365_joined %>% select(match_id, match_date, league, home_team, away_team, outcome, winning_team, closing_odds_outcome, 
                                          home_opening, home_closing, home_opening_minus_closing, 
                                          home_min, home_max, home_range,
                                          draw_opening, draw_closing, draw_opening_minus_closing, 
                                          draw_min, draw_max, draw_range,
                                          away_opening, away_closing, away_opening_minus_closing,
                                          away_min, away_max, away_range)

write.csv(bet365_joined, file = "bet365_outcome_features.csv")

## add match date
matches_all <- fread('/Users/nickvarberg/Desktop/School/Eulers-Men-Sports-Betting/bet365_matches.csv')
matches_all <- matches_all %>% select(match_id, match_datetime, league, home_team, away_team)
matches_all <- matches_all %>% mutate(match_date = str_trunc(match_datetime, 10, "right", ""))
matches_all <- matches_all %>% select(match_id, match_date, league, home_team, away_team)
bet365_dropped <- fread('/Users/nickvarberg/Desktop/School/Eulers-Men-Sports-Betting/bet365_outcome_features.csv')
