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
data_path <- "/Users/Erik/"
file_name <- ""

df <- data.frame(home_0=rnorm(5), home_1=rnorm(5), away_0=rnorm(5), away_1=rnorm(5)) %>% as_tibble()
df
for (time in seq(0,1)) {
  home_odds <- paste("home", as.character(time), sep="_")
  away_odds <- paste("away", as.character(time), sep="_")
  new_colm <- paste("home_away_diff", as.character(time), sep="_")
  df <- df %>% 
    mutate(UQ(rlang::sym(new_colm)) :=  UQ(rlang::sym(home_odds)) - UQ(rlang::sym(away_odds)))
}

num_differences = 6
72/num_differences
time_pts <- c(0, seq(1,72/num_differences)*num_differences-1)
tmp <- map_chr(time_pts, function(time) paste("home_away_diff", time, sep="_"))

df[tmp]

ml_

library(dplyr)
multipetalN <- function(df, n){
  varname <- paste0("petal.", n)
  df %>%
    mutate(!!varname := Petal.Width * n)
}

data(iris)
iris1 <- tbl_df(iris)
iris2 <- tbl_df(iris)
for(i in 2:5) {
  iris2 <- multipetalN(df=iris2, n=i)
}   
