library(tidyverse)
library(dplyr)
library(dslabs)
data(murders)

murders <- mutate(murders, rate = total / population * 10^5)

# instead of entering multiple parameters in the summarize() function,
# we can find the min, avg and max using just one function called quantile()
# 0th of the quantile is minimum, 0.5 is median and 1 is maximum
summary <- murders %>% filter(region == "West") %>% summarize(range = quantile(rate, c(0, 0.5, 1)))
summary

# however this returns a vector - to get a table, we can define a function as follows
min_median_max <- function(x){
  r <- quantile(x, c(0, 0.5, 1))
  data.frame(minimum = r[1], median = r[2], maximum = r[3])
}
murders %>% filter(region == "West") %>% summarize(min_median_max(rate))
