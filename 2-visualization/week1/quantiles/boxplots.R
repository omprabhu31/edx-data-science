library(tidyverse)
library(dslabs)
library(dplyr)
data(murders)

murders <- murders %>% mutate(rate = total / population * 10^5)

# histogram representation of murder rate
hist(murders$rate)

# qq representation of murder rate
p <- seq(0.05, 0.95, 0.05)
observed_quantiles <- quantile(murders$rate, p)
theoretical_quantiles <- qnorm(p, mean(murders$rate), sd(murders$rate))

qplot(theoretical_quantiles, observed_quantiles) + geom_abline()

# clearly, the distribution is not well represented by a normal approximation
# we can use a boxplot to better represent this data
boxplot(murders$rate)

# the solid line in the shaded area represents the median murder rate
# the top and bottom faces of the box represent the 3rd and 1st quartiles respectively
# the span of the dotted lines indicates the range of murder rate values (except outliers)
# the two separate points are outliers and plotted separately
