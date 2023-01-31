# confidence intervals are intervals with a specific chance of including the true parameter p
# the margin of error is approximately a 95% confidence interval

# for a confidence interval of size q, we solve for z = (1-q)/2
# to calculate the size of the confidence interval, we need to calculate the value of z
# for which Pr(-z <= Z <= z) equals the desired confidence

# for a 95% confidence interval
q <- 0.95
z <- abs(qnorm((1-q)/2)) # almost equal to 2 * standard dev i.e. margin of error

# Monte Carlo simulation of 95% confidence interval
p <- 0.45
N <- 1000
X <- sample(c(0,1), N, replace = TRUE, prob = c(p, 1-p))

X_hat <- mean(X)
se_hat <- sqrt(X_hat*(1-X_hat)/N)
c(X_hat - z*se_hat, X_hat + z*se_hat)

# additional stuff: geom_smooth for confidence interval plotting example
library(tidyverse)
data("nhtemp")
data.frame(year = as.numeric(time(nhtemp)), temperature = as.numeric(nhtemp)) %>%
  ggplot(aes(year, temperature)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Average Yearly Temperatures in New Haven")
