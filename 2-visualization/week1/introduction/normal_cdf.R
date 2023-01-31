# the normal distribution has a mathematically defined CDF
# it can be computed in R using the pnorm() function
# pnorm(a, avg, s) gives the value of the CDF F(a) for the normal distribution

library(tidyverse)
library(dslabs)
data(heights)
x <- heights %>% filter(sex=="Male") %>% pull(height)

# exact probability that a person's height is below 70.5 inches
1 - mean(x <= 70.5)

# estimation of the same using normal distribution
1 - pnorm(70.5, mean(x), sd(x))

# probabilities for data with range 1, containing an integer
mean(x <= 68.5) - mean(x <= 67.5)
mean(x <= 69.5) - mean(x <= 68.5)
mean(x <= 70.5) - mean(x <= 69.5)

# probabilities in normal approximation match well
pnorm(68.5, mean(x), sd(x)) - pnorm(67.5, mean(x), sd(x))
pnorm(69.5, mean(x), sd(x)) - pnorm(68.5, mean(x), sd(x))
pnorm(70.5, mean(x), sd(x)) - pnorm(69.5, mean(x), sd(x))

# probabilities in actual data over other ranges don't match normal approx as well
mean(x <= 70.9) - mean(x <= 70.1)
pnorm(70.9, mean(x), sd(x)) - pnorm(70.1, mean(x), sd(x))

# thus it makes sense to approximate the data by normal distribution for bins containing integer values
