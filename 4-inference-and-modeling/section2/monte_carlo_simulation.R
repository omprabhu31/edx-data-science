# we can run monte carlo simulations to compare with theoretical results assuming a value of p
p <- 0.45 # unknown p to estimate
N <- 1000 # poll size

se <- sqrt(p * (1-p) / N)

B <- 10000
x_hat <- replicate(B, {
  x <- sample(c(1,0), N, replace = TRUE, prob = c(p, 1-p))
  mean(x)
})

mean(x_hat)
sd(x_hat)

# histogram and qq-plots of the result
library(tidyverse)
library(gridExtra)

p1 <- data.frame(x_hat = x_hat) %>%
  ggplot(aes(x_hat)) +
  geom_histogram(binwidth = 0.005, color = "black")

p2 <- data.frame(x_hat = x_hat) %>%
  ggplot(aes(sample = x_hat)) +
  stat_qq(dparams = list(mean = mean(x_hat), sd = sd(x_hat))) +
  geom_abline() +
  ylab("X_hat") + 
  xlab("Theoretical normal")

grid.arrange(p1, p2, nrow = 1)
