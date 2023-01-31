library(tidyverse)
library(dslabs)
data(heights)
x <- heights %>% filter(sex == "Male") %>% .$height

# function to calculate proportion of male students no taller than a specific height
F <- function(a) mean(x <= a)

# probability that a male student is taller than 70.5 inches
1 - F(70)

# probability that a male student is between 60 and 65 inches
F(65) - F(60)
