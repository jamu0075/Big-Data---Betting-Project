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
library(car)
library(nortest)
setwd("/Users/Erik/Desktop/Eulers-Men-Sports-Betting/DataPreprocessing/")
data_path <- "/Users/Erik/Downloads/beat-the-bookie-odds-series-football-dataset/"

# Read in data
df <- fread(paste0(data_path, "games_full_odds.csv")) %>% as_tibble()

# Create outcome column {home, away, draw}
df <- df %>%
  mutate(outcome = ifelse(score_home>score_away, "home", ifelse(score_away>score_home, "away", "draw")))

# Filter data to leagues for which there are > # games
freq_by_league <- df %>%
  group_by(league) %>%
  summarise (n = n()) %>%
  mutate(freq = n / sum(n)) %>%
  arrange(desc(freq)) %>%
  mutate(cumsum_freq = cumsum(freq)) %>%
  mutate(cumsum_n = cumsum(n))
head(freq_by_league, n=20)
tail(freq_by_league, n=20)

game_num_thresh <- 200
leagues_many_games <- freq_by_league %>% filter(n>=game_num_thresh)

df <- df %>% filter(league %in% leagues_many_games$league)

# Construct tibble with all home, away, draw odds in 3 columns
all_home_odds <- c()
all_away_odds <- c()
all_draw_odds <- c()

for (time in seq(0,71)) {
  all_home_odds <- c(all_home_odds, pull(df, paste("home", as.character(time), sep="_")))
  all_away_odds <- c(all_away_odds, pull(df, paste("away", as.character(time), sep="_")))
  all_draw_odds <- c(all_draw_odds, pull(df, paste("draw", as.character(time), sep="_")))
}

all_odds <- tibble(
  'home_odds' = all_home_odds,
  'away_odds' = all_away_odds,
  'draw_odds' = all_draw_odds
)

# Make histograms of odds
quantiles <- function(x) {
  return(quantile(x=x, probs=c(0,0.001, 0.25,0.5,0.75,0.999,1)))
}

apply(all_odds, 2, quantiles)

home_away_diff = all_odds$home_odds - all_odds$away_odds
quantiles(home_away_diff)

x_range <- 30
prop_outside_range <- length(home_away_diff[abs(home_away_diff)>x_range ])/length(home_away_diff)

pdf('home_away_diff_hist.pdf')
ggplot() +
  geom_histogram(aes(x=home_away_diff, y=..density..)) +
  xlim(-x_range, x_range) +
  ggtitle(paste("Proportion of data outside (-", toString(x_range), ", ", toString(x_range), ") is ", toString(prop_outside_range), sep="")) 
dev.off()

pdf('qq_plot_home_away_diff.pdf')
# qqnorm(home_away_diff)
# qqline(home_away_diff, col = "steelblue", lwd = 2)
qqPlot(home_away_diff)
dev.off()

pdf('home_away_draw_hist.pdf')
all_odds %>%
  keep(is.numeric) %>%
  gather() %>%
  ggplot(aes(x=value, y=..density..)) + 
  facet_wrap(~key, scales="free") +
  geom_histogram() +
  xlim(0.9, 35)
dev.off()