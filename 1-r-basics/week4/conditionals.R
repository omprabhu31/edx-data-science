# general example if if-else statement
a <- 0
if (a != 0) {
  print(1/a)
} else {
  print("reciprocal of 0 does not exist.")
}

# an example to determine whether the minimum murder rate in the US is below 0.5
library(dslabs)
data(murders)
murder_rate <- murders$total / murders$population * 10^5
ind <- which.min(murder_rate)

if (murder_rate[ind] < 0.5) {
  print(murders$state[ind])
} else
{
  print("There is no state with a murder rate that low.")
}

# changing the value to 0.25 changes the result
if (murder_rate[ind] < 0.25) {
  print(murders$state[ind])
} else
{
  print("There is no state with a murder rate that low.")
}

# the ifelse() function works similar to the if-else conditional
a <- c(0, 1, 2, -4, 5)
ifelse (a > 0, 1/a, NA)

# the ifelse() function is also useful for replacing NA values
data(na_example)
sum(is.na(na_example))

no_nas <- ifelse(is.na(na_example), 0, na_example)
sum(is.na(no_nas))

# the any() and all() functions evaluate logical vectors
any(TRUE, TRUE, FALSE)
any(FALSE, FALSE, FALSE)

all(TRUE, TRUE, FALSE)
all(TRUE, TRUE, TRUE)
