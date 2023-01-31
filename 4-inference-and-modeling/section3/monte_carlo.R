# we can perform a monte carlo simulation to confirm that a 99% confidence
# interval contains p 99% of the time

p <- 0.45
N <- 1000
B <- 10000

q <- 0.99
z <- abs(qnorm((1-q)/2))

is_p_inside <- replicate(B, {
  X <- sample(c(0,1), N, replace = TRUE, prob = c(1-p, p))
  X_hat <- mean(X)
  se_hat <- sqrt(X_hat*(1-X_hat)/N)
  between(p, X_hat - z*se_hat, X_hat + z*se_hat) # 2*se is a rough approximation
})

mean(is_p_inside)
