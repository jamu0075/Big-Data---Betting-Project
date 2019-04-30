rm(list=ls())
gc()

###
# Assumption is that data has AT LEAST the following columns: (1) "league", (2) "outcome" (i.e., who won which is either home, away, or draw), and (2) home-, (3) away-, and (4) draw-odds at every hour up to 72 hours before the game. The form of the odds should be "`<home/away/draw>_<{0,1,...,71}>`" where 0 corresponds to closing odds (i.e., odds closest to game time) and 71 corresponds to opening odds (i.e., odds furthest from game time).
###

library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(purrr)
setwd("/Users/Erik/Desktop/Eulers-Men-Sports-Betting/DataPreprocessing/")
data_path <- "/Users/Erik/Downloads/beat-the-bookie-odds-series-football-dataset/"

# Read in data
df <- fread(paste0(data_path, "games_full_odds.csv")) %>% as_tibble()

# Create outcome column {home, away, draw}
df <- df %>%
  mutate(outcome = ifelse(score_home>score_away, "home", ifelse(score_away>score_home, "away", "draw")))
# df <- data.frame(home_0=rnorm(5), home_1=rnorm(5), away_0=rnorm(5), away_1=rnorm(5)) %>% as_tibble()
df

# Create columns for odds differences
for (time in seq(0,71)) {
  home_odds <- paste("home", as.character(time), sep="_")
  away_odds <- paste("away", as.character(time), sep="_")
  new_colm <- paste("home_away_diff", as.character(time), sep="_")
  df <- df %>% 
    mutate(UQ(rlang::sym(new_colm)) :=  UQ(rlang::sym(home_odds)) - UQ(rlang::sym(away_odds)))
}

# Filter data to leagues for which there are > games
freq_by_league <- df %>%
  group_by(league) %>%
  summarise (n = n()) %>%
  mutate(freq = n / sum(n)) %>%
  arrange(desc(freq)) %>%
  mutate(cumsum_freq = cumsum(freq)) %>%
  mutate(cumsum_n = cumsum(n))
head(freq_by_league, n=20)
tail(freq_by_league, n=20)

game_num_thresh <- 100
leagues_many_games <- freq_by_league %>% filter(n>=game_num_thresh)

df <- df %>% filter(league %in% leagues_many_games$league)

# Create a test and validation dataframe
percent_validation <- 0.15
nrows <- nrow(df)
# nrows <- 10
bool_index <- rep(TRUE, nrows)
valid_indices <- sample(seq(1,nrows), size=floor(percent_validation*nrows), replace=FALSE)

bool_index[valid_indices] <- FALSE
df_test <- df[bool_index,]
df_valid <- df[!bool_index,]

nrows - nrows*percent_validation
nrows*percent_validation

# Select a subset of the odds differences
num_diff = 18
ml_df_test <- construct_ml_df(df=df_test, num_differences=num_diff)
ml_df_valid <- construct_ml_df(df=df_valid, num_differences=num_diff)
colnames(ml_df_test)
colnames(ml_df_valid)

write.csv(ml_df_test, file = "ml_df_test.csv", row.names=FALSE)
write.csv(ml_df_valid, file = "ml_df_valid.csv", row.names=FALSE)


# Functions used
construct_ml_df <- function(df, num_differences) {
  # num_differences = 18
  # 72/num_differences
  time_pts <- c(0, seq(1,72/num_differences)*num_differences-1)
  odds <- map_chr(time_pts, function(time) paste("home_away_diff", time, sep="_"))
  odds <- c(odds, map_chr(time_pts, function(time) paste("draw", time, sep="_")))
  odds <- c(odds, map_chr(time_pts, function(time) paste("home", time, sep="_")))
  odds <- c(odds, map_chr(time_pts, function(time) paste("away", time, sep="_")))
  # time_pts
  # odds
  
  # Construct data frame for ML with outcome and predictors
  predictors <- c("league", "home_team", "away_team", odds)
  
  ml_df <- df[c("outcome", predictors)]
  
  return(ml_df)
}

# Test code
num_differences = 18
72/num_differences
time_pts <- c(0, seq(1,72/num_differences)*num_differences-1)
odds <- map_chr(time_pts, function(time) paste("home_away_diff", time, sep="_"))
odds <- c(odds, map_chr(time_pts, function(time) paste("draw", time, sep="_")))
odds <- c(odds, map_chr(time_pts, function(time) paste("home", time, sep="_")))
odds <- c(odds, map_chr(time_pts, function(time) paste("away", time, sep="_")))
time_pts
odds

# Construct data frame for ML with outcome and predictors
predictors <- c("league", "home_team", "away_team", odds_diffs)

ml_df <- df[c("outcome", predictors)]

write.csv(ml_df, file = "ml_test_df.csv", row.names=FALSE)

##################################
# Data exploration
head(df$home_away_diff_0)
head(df$home_0)
head(df$away_0)


