library(data.table)
library(dplyr)
library(tidyr)
data_path <- "/Users/nickvarberg/Downloads/beat-the-bookie-odds-series-football-dataset/"

#### 2 filenames and change leaguetable1/2####
series      <- fread(paste0(data_path, "odds_series_b.csv"))
series_matches <- fread(paste0(data_path, "odds_series_b_matches.csv"))
series_matches <- series_matches %>% filter(match_id %in% series$match_id)
series <- full_join(series, series_matches, by = 'match_id')
rm(series_matches)
leaguetable2 <- series %>% select(match_id, league)
rm(series)
gc()
#
leaguetable <- full_join(leaguetable1, leaguetable2, by = "match_id")
leaguetable <- leaguetable %>% mutate("league" = league.x) %>% select(match_id, league)
leaguetable <- leaguetable %>%
  group_by(league) %>%
  mutate("n_games" = n())
leaguetable <- leaguetable %>%
  filter(n_games >= 100) %>%
  ungroup()
rm(leaguetable1, leaguetable2)
write.csv(leaguetable, file = "leaguetable.csv")

#### ####
# load just each book, filter to rows with no missing, load next book and add unique rows
bookkeeper = 9
cols = c(1:5, (6+(bookkeeper-1)*num):(6+bookkeeper*num - 1))
series_b = fread(paste0(data_path, "odds_series_b.csv"), select = cols)
series = fread(paste0(data_path, "odds_series.csv"), select = cols)
series_all = full_join(series, series_b)
rm(series)
rm(series_b)
gc()
series_all <- series_all %>%
  drop_na()
colnms = colnames(series_all)
for (i in 6:length(colnms)){
  colnms[i] = paste0(substr(colnms[i], 1, 5), toString((i-6)%%72))
}
colnames(series_all) = colnms


for (bookkeeper in c(1:6,8,10:32)){  # already did 7 and started with 9
  cols = c(1:5, (6+(bookkeeper-1)*num):(6+bookkeeper*num - 1))
  series_b = fread(paste0(data_path, "odds_series_b.csv"), select = cols)
  series = fread(paste0(data_path, "odds_series.csv"), select = cols)
  series_all_book = full_join(series, series_b)
  rm(series)
  rm(series_b)
  gc()
  series_all_book <- series_all_book %>%
    drop_na() %>%
    filter(!match_id %in% series_all$match_id)
  colnms = colnames(series_all_book)
  for (i in 6:length(colnms)){
    colnms[i] = paste0(substr(colnms[i], 1, 5), toString((i-6)%%72))
  }
  colnames(series_all_book) = colnms
  series_all = full_join(series_all, series_all_book)
  print(bookkeeper)
}
write.csv(series_all, file = "series_with_odds.csv")
