# Monte Carlo simulations model the probability of different outcomes by
# repeating a random process for a large enough number of times such that 
# the results would be practically equivalent to repeating it forever

# example: urn with 2 red beads and 3 blue beads
beads <- rep(c('red', 'blue'), times=c(2, 3))
beads

# the sample() function draws random outcomes from a set of options
# note that it does NOT account for replacement by default
sample(beads, 1)
sample(beads, 5)
# sample(beads, 6) - error
sample(beads, 6, replace = TRUE)

# performing a simulation for 10000 draws
n <- 10000
events <- replicate(n, sample(beads, 1)) # draw 1 bead, n times
tab <- table(events) # make a table of outcome counts
tab
prop.table(tab) # make a table of outcome proportions

# how many experiments are enough? we can plot the monte carlo results 
# for various values of B and see where it stabilizes
B <- 10^seq(1, 5, len = 100)
compute_prob <- function(B, n = 22) {
  same_day <- replicate(B, {
    bdays <- sample(1:365, n, replace = TRUE)
    any(duplicated(bdays))
  })
  mean(same_day)
}

prob <- sapply(B, compute_prob)
lines(log10(B), prob)
