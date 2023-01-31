# interest rates for loans are set using the probability of loan defaults (approx 2%)
# to calculate a rate that minimizes the risk of losing money

# let's initially start with a 0% interest rate
n <- 1000
loss_per_foreclosure <- -200000
p <- 0.02
defaults <- sample(c(0,1), n, replace = TRUE, prob = c(1-p, p))
sum(defaults * loss_per_foreclosure)

# here is a monte carlo simulation for the same
library(tidyverse)
B <- 10000
losses <- replicate(B, {
  defaults <- sample(c(0,1), n, replace = TRUE, prob = c(1-p, p))
  sum(defaults * loss_per_foreclosure)
})

data.frame(losses_in_millions = losses / 10^6) %>% ggplot(aes(losses_in_millions)) +
  geom_histogram(binwidth = 0.6, col = "black")

# since the histogram closely resembles that of a normal approximation, we can use CLT
# expected value and std error for 1000 loans
expected_value <- n * (p*loss_per_foreclosure + (1-p)*0)
std_error <- sqrt(n) * abs(loss_per_foreclosure) * sqrt(p*(1-p))

# let us try to set an interest rate where we break even with the losses
# in short, we want to make the expected value equal to zero (assume $180000 loan)
x <- -loss_per_foreclosure * p / (1-p)
interest_rate <- x / 180000 * 100
interest_rate

# let us say we want the probability of losing money to be a maximum of 1%
l <- loss_per_foreclosure
z <- qnorm(0.01)
x <- -l * (n*p - z*sqrt(n*p*(1-p))) / (n*(1-p) + z*sqrt(n*p*(1-p)))

interest_rate <- x / 180000 * 100
interest_rate

expected_profit <- l*p + x*(1-p)
expected_profit

expected_total_profit <- n * expected_profit
expected_total_profit

# monte carlo simulation for 1% probability of losing money
B <- 10000
profit <- replicate(B, {
  draws <- sample(c(x, loss_per_foreclosure), n, prob = c(1-p, p), replace= TRUE)
  sum(draws)
})

mean(profit)
mean(profit < 0) # chance of losing money
