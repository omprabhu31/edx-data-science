library(data.table)
library(tidyverse)
library(dplyr)
library(dslabs)
data(heights)

heights <- setDT(heights)

# summarizing in dplyr vs data.table
heights %>% summarize(average = mean(height),
                      std_dev = sd(height))
heights[, .(average = mean(height), std_dev = sd(height))]

# filter and summarizing in dplyr vs data.table
heights %>% filter(sex == "Female") %>% summarize(average = mean(height),
                      std_dev = sd(height))
heights[sex == "Female", .(average = mean(height), std_dev = sd(height))]

# grouping and summarizing in dplyr vs data.table
heights %>% group_by(sex) %>% summarize(average = mean(height),
                                                  std_dev = sd(height))
heights[, .(average = mean(height), std_dev = sd(height)), by = sex]

# multiple summaries in dplyr vs data.table
median_min_max = function(x){
  r <- quantile(x, c(0.5, 0, 1))
  data.frame(median = r[1], minimum = r[2], maximum = r[3])
}

heights %>% summarize(median_min_max(height))
heights[, median_min_max(height)]
