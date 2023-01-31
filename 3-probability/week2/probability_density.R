# the probability density function is defined for continuous data at a point
# the CDF is the integral of PDF at all real values of x up to a specific value

# we can use dnorm() to plot the density curve for the normal distribution
library(tidyverse)
library(dslabs)
data(heights)
x <- heights %>% filter(sex == "Male") %>% .$height

data.frame(x, f = dnorm(x, mean(x), sd(x))) %>% ggplot(aes(x, f)) + geom_line()
