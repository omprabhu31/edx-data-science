library(tidyverse)
library(dslabs)
data(heights)

p <- heights %>% filter(sex == "Male") %>% ggplot(aes(x = height))

p + geom_histogram()