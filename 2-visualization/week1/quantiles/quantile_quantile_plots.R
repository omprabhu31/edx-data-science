# a QQ plot can be used to check whether a distribution is well approximated
# by a normal distribution
library(tidyverse)
library(dslabs)
library(dplyr)
data(heights)

male <- heights %>% filter(sex == "Male")

p <- seq(0.05, 0.95, 0.05)
observed_quantiles <- quantile(male$height, p)
theoretical_quantiles <- qnorm(p, mean(male$height), sd(male$height))

plot(theoretical_quantiles, observed_quantiles)
abline(0,1)

# we can also plot a scaled qq plot
z <- scale(male$height)
observed_quantiles <- quantile(z, p)
theoretical_quantiles <- qnorm(p)

plot(theoretical_quantiles, observed_quantiles)
abline(0,1)
