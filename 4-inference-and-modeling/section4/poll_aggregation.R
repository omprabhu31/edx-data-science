# simulating individual polls for 2012 presidential election
d <- 0.039 # actual election day spread
Ns <- c(1298, 533, 1342, 897, 774, 254, 812, 324, 1291, 1056, 2172, 516) # sample sizes
p <- (d+1)/2

# calculate 95% confidence intervals of the spread for each poll
confidence_intervals <- sapply(Ns, function (N) {
  X <- sample(c(0,1), N, replace = TRUE, prob = c(1-p,p))
  X_hat <- mean(X)
  se_hat <- sqrt(X_hat*(1-X_hat)/N)
  2*c(X_hat, X_hat - 2*se_hat, X_hat + 2*se_hat) - 1
})

# generate data frame from results
polls <- data.frame(poll = 1:ncol(confidence_intervals),
                    t(confidence_intervals), sample_size = Ns)
names(polls) <- c("poll", "estimate", "low", "high", "sample_size")
polls

# a much narrower confidence interval can be obtained by using the combined spread of polls
library(tidyverse)
d_hat <- polls %>% summarize(avg = sum(estimate * sample_size)/sum(sample_size)) %>%
  pull(avg)
p_hat <- (1+d_hat)/2
margin_of_error <- 2*1.96*sqrt(p_hat*(1-p_hat)/sum(polls$sample_size))

# observe that the confidence interval no longer includes 0
round(d_hat*100, 1)
round(margin_of_error*100, 1)
