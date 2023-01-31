library(tidyverse)
library(dslabs)
data(gapminder)
head(gapminder)

ds_theme_set()

# time series plots are preferred for analysing trends between a few selected variables
# eg: fertility rate trend in the United States
gapminder %>% filter(country == "United States") %>% ggplot(aes(year, fertility)) +
  geom_point()

# the geom_line() function helps better analyse the trends
gapminder %>% filter(country == "United States") %>% ggplot(aes(year, fertility)) +
  geom_line()

# we can also group by variable to compare the plots
gapminder %>% filter(country %in% c("South Korea", "Germany")) %>% 
  ggplot(aes(year, fertility, col = country)) + geom_line()

# in most cases, data labels are preferred over legends
countries <- c("South Korea", "Germany")
labels <- data.frame(country = countries, x = c(1975, 1965), y = c(60, 72))

gapminder %>% filter(country %in% countries) %>%
  ggplot(aes(year, life_expectancy, col = country)) + 
  geom_line() + geom_text(data = labels, aes(x, y, label = country), size = 5) +
  theme(legend.position = "none")
