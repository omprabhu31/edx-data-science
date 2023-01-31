library(tidyverse)
library(dslabs)
data(gapminder)

ds_theme_set()

gapminder <- gapminder %>% mutate(dollars_per_day = gdp/population/365)
head(gapminder)

# when ordered by region, we can very clearly see that there is indeed a west vs
# the rest dichotomy
gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% 
  mutate(region = reorder(region, dollars_per_day, FUN = median)) %>%
  ggplot(aes(dollars_per_day, region)) +
  geom_point() + scale_x_continuous(trans = "log2")

# grouping by regions in boxplot
gapminder <- gapminder %>% mutate(group = case_when(
  region %in% c("Western Europe", "Northern Europe","Southern Europe", 
                "Northern America", 
                "Australia and New Zealand") ~ "West",
  region %in% c("Eastern Asia", "South-Eastern Asia") ~ "East Asia",
  region %in% c("Caribbean", "Central America", 
                "South America") ~ "Latin America",
  continent == "Africa" & 
    region != "Northern Africa" ~ "Sub-Saharan",
  TRUE ~ "Others"))

gapminder %>% filter(year == 1970 & !is.na(gdp)) %>% 
  mutate(group = reorder(group, dollars_per_day, FUN = median)) %>%
  ggplot(aes(group, dollars_per_day, fill = group)) +
  geom_boxplot(show.legend = FALSE) + scale_y_continuous(trans = "log2")
