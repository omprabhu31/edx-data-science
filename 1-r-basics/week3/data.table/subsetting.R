library(data.table)
library(tidyverse)
library(dplyr)
library(dslabs)
data(murders)

murders <- setDT(murders)
murders[, rate := total / population * 100000]

# the data.table package is in many ways better at subsetting than dplyr
filter(murders, rate <= 0.71)
murders[rate <= 0.71]

# the filter and select functions can also be combined into one command
murders %>% filter(rate <= 0.71) %>% select(state, region)
murders[rate <= 0.71, .(state, region)]
