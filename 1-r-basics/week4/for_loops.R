# for loops perform the same task over and over again while changing the variable
# let us compute the sum of the first n natural numbers
compute_s_n <- function(n) {
  x <- 1:n
  sum(x)
}

# using a for loop to calculate this sum for n = 1, ..., 25
m <- 25
s_n <- vector(length = m)
for (i in 1:m) {
  s_n[i] <- compute_s_n(i)
}

s_n
