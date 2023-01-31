library(tidyverse)
library(dslabs)
data(gapminder)

ds_theme_set()

gapminder <- gapminder %>% mutate(dollars_per_day = gdp/population/365)
head(gapminder)

west <- c("Western Europe", "Northern Europe", "Northern America", "Australia and New Zealand")

country_list1 <- gapminder %>% filter(year == 1970 & !is.na(gdp)) %>%
  pull(country)
country_list2 <- gapminder %>% filter(year == 2010 & !is.na(gdp)) %>%
  pull(country)
country_list <- intersect(country_list1, country_list2)

# for a regular smooth density plot, area under each curve would be 1
# this would not reflect the number of developing vs western countries
gapminder %>%
  filter(year == 1970 & country %in% country_list) %>%
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>% group_by(group) %>%
  summarize(n = n()) %>% knitr::kable()

p <- gapminder %>% filter(year == 1970 & country %in% country_list) %>%
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>% 
  ggplot(aes(dollars_per_day, y = ..count.., fill = group)) +
  scale_x_continuous(trans = "log2")
p + geom_density(alpha = 0.2, bw = 0.75) + facet_grid(year~.)

# adding new region groups with case_when
gapminder <- gapminder %>% mutate(group = case_when(
  region %in% west ~ "West",
  region %in% c("Eastern Asia", "South-Eastern Asia") ~ "East Asia",
  region %in% c("Caribbean", "Central America", "South America") ~ "Latin America",
  continent == "Africa" & region != "Northern Africa" ~ "Sub-Saharan Africa",
  TRUE ~ "Others"))
#reorder factor levels
gapminder <- gapminder %>% 
  mutate(group= factor(group, levels = c("Others", "Latin America", "East Asia", "Sub-Saharan Africa", "West")))

# stacked density plot
p <- gapminder %>% filter(year %in% c(1970, 2010) & country %in% country_list) %>%
  ggplot(aes(dollars_per_day, fill = group)) +
  scale_x_continuous(trans = "log2")
p + geom_density(alpha = 0.2, bw = 0.75, position = "stack") + facet_grid(year~.)

# this plot, however, does not take into account the populations of countries
gapminder %>% filter(year %in% c(1970, 2010) & country %in% country_list) %>% 
  group_by(year) %>% mutate(weight = population / sum(population*2)) %>%
  ungroup() %>% ggplot(aes(dollars_per_day, fill = group, weight = weight)) + 
  scale_x_continuous(trans = "log2") + 
  geom_density(alpha = 0.2, bw = 0.75, position = "stack") + facet_grid(year~.)
