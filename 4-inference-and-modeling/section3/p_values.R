# null hypothesis -> there is no effect
# in the case of blue and red beads in an urn, the null hypothesis is that the 
# spread is 0, or p = 0.5

# the p-value is the probability of detecting an effect of a certain size or larger 
# when the null hypothesis is true

# let's say we want to find the probability that in an urn of 100 beads, 52 beads are blue
N <- 100
p_null <- 0.5
p_desired <- 0.52

spread <- p_desired - p_null
z <- sqrt(N)*spread/sqrt(p_null * (1 - p_null))

1 - (pnorm(z) - pnorm(-z)) # p-value

# note now the term being subtracted from 1 is the confidence interval itself
# thus, we can say that if a given confidence interval does not include our desired value
# then the p-value must be smaller than 0.5
