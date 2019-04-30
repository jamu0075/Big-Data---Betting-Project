rm(list=ls())
gc()

library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
data_path <- "/Users/Erik/Downloads/beat-the-bookie-odds-series-football-dataset/"

# Load data
# series and series_b have match_id match_date, match_time, score_home, score_away, and odds for different bookies at different times
# series_matches and series_b_matches have match_id, league, home_team (name), away_team (name), score, detailed_score, match_datetime
series <- fread(paste0(data_path, "odds_series.csv"))
series_b <- fread(paste0(data_path, "odds_series_b.csv"))
series_all <- full_join(series, series_b) %>% as_tibble()
saveRDS(series_all, "series_all.rds")
rm(series, series_b)
gc()

series_matches <- fread(paste0(data_path, "odds_series_matches.csv"))
series_b_matches <- fread(paste0(data_path, "odds_series_b_matches.csv"))
matches_all <- full_join(series_matches, series_b_matches) %>% as_tibble()

# Use only those games for which we have betting data
matches_all <- matches_all %>% filter(match_id %in% series_all$match_id)
saveRDS(matches_all, "matches_all.rds")
rm(series_matches, series_b_matches)
gc()

# Explore data by league
freq_by_league <- final_df %>%
  group_by(league) %>%
  summarise (n = n()) %>%
  mutate(freq = n / sum(n)) %>%
  arrange(desc(freq)) %>%
  mutate(cumsum_freq = cumsum(freq)) %>%
  mutate(cumsum_n = cumsum(n))
head(freq_by_league, n=20)
tail(freq_by_league, n=20)

# Add "outcome" column to data based on who won {"home", "draw", or "away"}
series_all <- series_all %>% mutate(outcome = ifelse(score_home>score_away, "home", ifelse(score_away>score_home, "away", "draw")))


df <- fread(paste0(data_path, "games_full_odds.csv")) %>% as_tibble()
num_time_pts <- 8
time_pts <- seq(1,as.integer(72/num_time_pts))


bet365 <- fread("series_premier_league_bet365.csv")
bet365 <- bet365 %>%
  mutate(outcome = if_else(score_home >= score_away, if_else(score_home == score_away, "draw", "home"), "away")) %>%
  mutate(home_opening = "home_71") %>%
  mutate(home_closing = "home_0") %>%
  mutate(draw_opening = "draw_71") %>%
  mutate(draw_closing = "draw_0") %>%
  mutate(away_opening = "away_71") %>%
  mutate(away_closing = "away_0") %>%
  mutate(home_opening_minus_closing = home_opening - home_closing) %>%
  mutate(draw_opening_minus_closing = draw_opening - draw_closing) %>%
  mutate(away_opening_minus_closing = away_opening - away_closing) %>%
  mutate(closing_odds_outcome = if_else(outcome %in% c("home", "draw"), (if_else(outcome == "home", home_closing, draw_closing)), away_closing))

select(tmp, score_home, score_away, outcome)
head(freq_by_league, 20)
tail(freq_by_league, 10)
series_all
freq_by_league[30:60,]
any(tmp$league=="USA: MLS")
tmp <- freq_by_league %>% filter(n>=200)
tail(freq_by_league)
dim()
head(series)
colnames(series)
colnames(series_matches)
colnames(series_b)
colnames(series_b_matches)
