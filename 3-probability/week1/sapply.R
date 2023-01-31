compute_prob <- function(students, n = 10000) {
  same_day <- replicate(n, {
    bdays <- sample(1:365, students, replace=TRUE)
    any(duplicated(bdays))
  })
  mean(same_day)
}

# the sapply() function allows any function to be applied element-wise to a vector
# note than some functions like sqrt() and * do this by default
n <- seq(1, 60)
prob <- sapply(n, compute_prob)

# we can also compute the exact probability as follows
exact_prob <- function(n) {
  prob_unique <- seq(365, 365 - n + 1)/365
  1 - prod(prob_unique)
}

eprob <- sapply(n, exact_prob)

plot(n, prob)
lines(n, eprob, col = "red")