library(tidyverse)
library(dslabs)
data(gapminder)

ds_theme_set()

gapminder <- gapminder %>% mutate(dollars_per_day = gdp/population/365)
head(gapminder)

# histogram of the west vs the developing world
west <- c("Western Europe", "Northern Europe", "Northern America", "Australia and New Zealand")
# facet by west vs developing
gapminder %>% filter(year %in% c(1970) & !is.na(gdp)) %>%
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>%
  ggplot(aes(dollars_per_day)) + 
  geom_histogram(binwidth = 1, color = "black") + 
  scale_x_continuous(trans = "log2") +
  facet_grid(~group)

# facet by year and west vs developing
gapminder %>% filter(year %in% c(1970, 2010) & !is.na(gdp)) %>%
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>%
  ggplot(aes(dollars_per_day)) + 
  geom_histogram(binwidth = 1, color = "black") + 
  scale_x_continuous(trans = "log2") +
  facet_grid(year~group)

# notice that data is available for more countries in 2010 than in 1970
# let's plot the histogram for only the countries common between 2010 and 1970
country_list1 <- gapminder %>% filter(year == 1970 & !is.na(gdp)) %>%
  pull(country)
country_list2 <- gapminder %>% filter(year == 2010 & !is.na(gdp)) %>%
  pull(country)
country_list <- intersect(country_list1, country_list2)

gapminder %>% filter(year %in% c(1970, 2010) & !is.na(gdp) & country %in% country_list) %>%
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>%
  ggplot(aes(dollars_per_day)) + 
  geom_histogram(binwidth = 1, color = "black") + 
  scale_x_continuous(trans = "log2") +
  facet_grid(year~group)

# boxplots of income in the west vs developing world, 1970 and 2010
gapminder %>% filter(year %in% c(1970, 2010) & !is.na(gdp) & country %in% country_list) %>%
  mutate(region = reorder(region, dollars_per_day, FUN = median)) %>%
  ggplot(aes(region, dollars_per_day, fill = continent)) + 
  geom_boxplot() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_y_continuous(trans = "log2") +
  facet_grid(year~.)

# plotting boxplots side by side
gapminder %>% filter(year %in% c(1970, 2010) & !is.na(gdp) & country %in% country_list) %>%
  mutate(region = reorder(region, dollars_per_day, FUN = median)) %>%
  ggplot(aes(region, dollars_per_day, fill = factor(year))) + 
  geom_boxplot() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_y_continuous(trans = "log2")
