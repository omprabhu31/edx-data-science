library(tidyverse)
library(dslabs)
data(gapminder)

ds_theme_set()

gapminder <- gapminder %>% mutate(dollars_per_day = gdp/population/365)
head(gapminder)

# boxplot of GDP by region in 1970
gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% 
  ggplot(aes(region, dollars_per_day)) + geom_boxplot()

# boxplot with rotated x axis labels
gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% 
  ggplot(aes(region, dollars_per_day)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# the reorder() function helps reorder factors according to another category
gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% 
  mutate(region = reorder(region, dollars_per_day, FUN = median)) %>%
  ggplot(aes(region, dollars_per_day, fill = continent)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# transforming y axis value by log2
gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% 
  mutate(region = reorder(region, dollars_per_day, FUN = median)) %>%
  ggplot(aes(region, dollars_per_day, fill = continent)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_y_continuous(trans = "log2")

# add points layer to show individual values
gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% 
  mutate(region = reorder(region, dollars_per_day, FUN = median)) %>%
  ggplot(aes(region, dollars_per_day, fill = continent)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_y_continuous(trans = "log2") +
  geom_point(show.legend = FALSE)
