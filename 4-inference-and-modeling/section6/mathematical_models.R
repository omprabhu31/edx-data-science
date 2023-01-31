library(tidyverse)
library(dslabs)
polls <- polls_us_election_2016 %>%
  filter(state == "U.S." & enddate >= "2016-10-31" &
           (grade %in% c("A+", "A", "A-", "B+") | is.na(grade))) %>%
  mutate(spread = rawpoll_clinton/100 - rawpoll_trump/100)

one_poll_per_pollster <- polls %>% group_by(pollster) %>%
  filter(enddate == max(enddate)) %>%
  ungroup()

results <- one_poll_per_pollster %>%
  summarize(avg = mean(spread), se = sd(spread)/sqrt(length(spread))) %>%
  mutate(start = avg - 1.96*se, end = avg + 1.96*se)

# if we collect several polls with measures spreads X_i, these random variables have
# expected value d and standard error 2*sqrt(p*(1-p)/N)

# to account for random errors, pollster to pollster variability and general bias,
# we represent the measurements as X_ij = d + b + h_i + epsilon_ij (i -> pollsters, j -> polls)
# where: d = actual spread of the election
#        b = general bias affecting all pollsters (affects the posterior standard error)
#        h_i = house effect for i-th pollster
#        epsilon_ij = random error associated with i,j-th poll

# thus, the new sample average is X_hat = d + b + sum(X_i)/N
# and new standard deviation is sqrt(sigma^2/N + sigma_b^2)

# simulated data with X_j = d + epsilon_j (one pollster, multiple polls)
J <- 6
N <- 2000
d <- 0.021
p <- (d+1)/2

X <- d + rnorm(J, 0, sqrt(p*(1-p)/N))
X

# simulated data with X_ij = d + epsilon_ij (multiple pollsters, multiple polls)
J <- 6
I <- 5
N <- 2000
d <- 0.021
p <- (d+1)/2

X <- sapply(1:I, function(i){
  d + rnorm(J, 0, sqrt(p*(1-p)/N))
})
X

# simulated data with X_ij = d + h_i + epsilon_ij
J <- 6
I <- 5
N <- 2000
d <- 0.021
p <- (d+1)/2
h <- rnorm(I, 0, 0.025) # sd based on historical data

X <- sapply(1:I, function(i){
  d + h[i] + rnorm(J, 0, sqrt(p*(1-p)/N))
})
X

# calculating probability of d > 0 with general bias
mu <- 0
tau <- 0.035
sigma <- sqrt(results$se^2 + 0.025^2)
Y <- results$avg
B <- sigma^2/(sigma^2 + tau^2)

posterior_mean <- B*mu + (1-B)*Y
posterior_se <- sqrt(1/(1/sigma^2 + 1/tau^2))

1 - pnorm(0, posterior_mean, posterior_se)
