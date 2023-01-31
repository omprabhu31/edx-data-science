library(tidyverse)
library(dslabs)
data(gapminder)
head(gapminder)

ds_theme_set()

# life expectancy vs fertility rate in 1962
filter(gapminder, year == 1962) %>% 
  ggplot(aes(fertility, life_expectancy, col = continent)) + 
  geom_point()

# faceting by continent and year - comparing 1962 vs 2012
filter(gapminder, year %in% c(1962, 2012)) %>% 
  ggplot(aes(fertility, life_expectancy, col = continent)) + 
  geom_point() + facet_grid(continent~year)

# faceting by year only - comparing 1962 vs 2012
filter(gapminder, year %in% c(1962, 2012)) %>% 
  ggplot(aes(fertility, life_expectancy, col = continent)) + 
  geom_point() + facet_grid(~year)

# faceting over multiple years to observe trend
years <- c(1962, 1970, 1980, 1990, 2000, 2012)
continents <- c("Europe", "Asia")
filter(gapminder, year %in% years & continent %in% continents) %>% 
  ggplot(aes(fertility, life_expectancy, col = continent)) + 
  geom_point() + facet_wrap(~year)
