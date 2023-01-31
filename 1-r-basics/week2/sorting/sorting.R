library(dslabs)
data(murders)

# the sort() function sorts a vector in increasing order
x <- c(31, 4, 15, 92, 65)
x
sort(x)

# the order() function produces the indices needed to obtain the sorted vector
order(x) # 4 is the 2nd element in x, 15 is the 3rd element in x, and so on

# the rank() function returns the ranks of the elements in the original vector
rank(x) # 31 is the 3rd smallest, 4 is the smallest, and so on

# max() returns max value, which.max() returns index of max value, same for min
max(x)
which.max(x)

min(x)
which.min(x)

# using the murders dataset as a more practical example
no_of_murders <- sort(murders$total)
no_of_murders

# obtaining the state and abbreviation in increasing order of total murders
murder_rate <- order(murders$total)
state_order <- murders$state[murder_rate]
abb_order <- murders$abb[murder_rate]

state_order
abb_order

# generate a labeled list of increasing order of murders
names(no_of_murders) = state_order
no_of_murders

# to create a dataframe that orders states from least populous to most
states <- murders$state
ranks <- rank(murders$population)
ind <- order(murders$population)

pop_order <- data.frame(states = states[ind])
pop_order