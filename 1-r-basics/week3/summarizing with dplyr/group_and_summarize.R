library(tidyverse)
library(dplyr)
library(dslabs)
data(murders)

murders <- mutate(murders, rate = total / population * 10^5)

# the function group_by() can be used to split data into groups
# the result is a grouped data frame
murders %>% group_by(region)

# grouping and summarizing is one of the most useful techniques
murders %>% group_by(region) %>% summarize(median_rate = median(rate))

# let's try this for a different dataset
data(heights)
heights %>% group_by(sex) %>% summarize(avg_height = mean(height),
                                        std_dev = sd(height))
