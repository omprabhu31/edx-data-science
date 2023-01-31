library(dslabs)
data(murders)

murder_rate <- murders$total / murders$population * 100000

# the which() function can be used to find out which element satisfies a particular condition
ind <- which(murders$state == "California")
murder_rate[ind]

# the match() function can be used to find out which indices of the second vector match 
# the entries of the first vector
ind <- match(c("New York", "Florida", "Texas"), murders$state)
ind
murder_rate[ind]

# the %in% function can be used to find whether or not each element of the first vector
# is in the second vector
c("Mumbai", "Japan", "Washington", "North Dakota", "Austin") %in% murders$state

# there is a connection between match and %in% through which
match(c("New York", "Florida", "Texas"), murders$state)
which(murders$state %in% c("New York", "Florida", "Texas"))
