# define random variable to be 1 if blue, 0 otherwise
beads <- rep(c("red", "blue"), c(2, 3))

ifelse(sample(beads, 1) == "blue", 1, 0)
ifelse(sample(beads, 1) == "blue", 1, 0)
ifelse(sample(beads, 1) == "blue", 1, 0)
