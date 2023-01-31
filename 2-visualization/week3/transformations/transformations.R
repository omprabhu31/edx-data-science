library(tidyverse)
library(dslabs)
data(gapminder)

ds_theme_set()

gapminder <- gapminder %>% mutate(dollars_per_day = gdp/population/365)
head(gapminder)

# histogram of dollars per day in 1970
gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% ggplot(aes(dollars_per_day)) + 
  geom_histogram(binwidth = 1, color = "black")

# same histogram with log2 scaled data
gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% ggplot(aes(log2(dollars_per_day))) + 
  geom_histogram(binwidth = 1, color = "black")

# same histogram with x axis scaled to log2
gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% ggplot(aes(dollars_per_day)) + 
  geom_histogram(binwidth = 1, color = "black") + scale_x_continuous(trans = "log2")
