library(tidyverse)
library(dplyr)
library(dslabs)
data(murders)

murders <- mutate(murders, rate = total / population * 10^5)

# minimum, maximum and median are some useful summary statistics
# the summarize() function is an easy way to compute summary stats
summary <- murders %>% filter(region == "West") %>% summarize(minimum = min(rate),
                                                              median = median(rate),
                                                              maximum = max(rate))
summary

# components of the summary can be accessed using the $ accessor
summary$median

# we can use it to find an average murder rate weighted by population
us_murder_rate <- murders %>% summarize(rate = sum(total) / sum(population) * 10^5)
us_murder_rate

# instead of representing it as a dataframe, we can use pull() to extract the numeric value
us_murder_rate <- murders %>% summarize(rate = sum(total) / sum(population) * 10^5) %>% pull(rate)
us_murder_rate

# the dot . can be used as a placeholder for the data being passed through the pipe
# eg:- murders %>% summary(-----) %>% .$rate, here the . represents the dataframe containing the summary
murders %>% summarize(rate = sum(total) / sum(population) * 10^5) %>% .$rate
