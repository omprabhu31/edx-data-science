# consider the distribution of 12 blue beads and 13 red beads in an urn
# a blue bead corresponds to 1 and red bead corresponds to zero
# since X is the sum of draws from an urn, its distribution is approximately normal

X_hat <- 12 / (13 + 12)

# since we don't know p, we can use a plug-in estimate to calculate the standard error
se <- sqrt(X_hat * (1 - X_hat) / 25)

# probability of the sum being within 1% of the expected value
pnorm(0.01/se) - pnorm(-0.01/se)

# the margin of error is equal to 2 standard errors (containing 95% of values)
# clearly a poll with only 25 people is not useful due to the large margin of error
2 * se
