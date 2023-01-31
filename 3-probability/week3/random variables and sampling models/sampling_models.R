# sampling models model random behaviour of a process as the sampling of draws from an urn
# consider a casino that wishes to add a roulette wheel - calculate the chances of loss
# we consider the game of only choosing between reds and blacks

# sampling model 1: define urn, then sample
color <- rep(c("Black", "Red", "Green"), c(18, 18, 2))
n <- 1000 # number of people playing
X <- sample(ifelse(color == "Red", -1, 1), n, replace = TRUE)
X[1:10]

# sampling model 2: define urn inside sample function itself by noting probabilities
X <- sample(c(-1, 1), n, replace = TRUE, prob = c(9/19, 10/19))
S <- sum(X)
S

# to generate the probability distribution of the random variable, we can run a 
# monte carlo simulation to find the chance of losing money
B <- 10000
n <- 1000
S <- replicate(B, {
  X <- sample(c(-1, 1), n, replace = TRUE, prob = c(9/19, 10/19))
  sum(X)
})
mean(S < 0) # probability of making a loss

# check if the monte carlo distribution agrees with normal density curve
library(tidyverse)
s <- seq(min(S), max(S), length = 100)
normal_density <- data.frame(s = s, f = dnorm(s, mean(S), sd(S)))
data.frame(S = S) %>% ggplot(aes(S, ..density..)) +
  geom_histogram(color = 'black', binwidth= 10) +
  ylab('Probability') +
  geom_line(data = normal_density, mapping = aes(s, f), color = 'blue')
