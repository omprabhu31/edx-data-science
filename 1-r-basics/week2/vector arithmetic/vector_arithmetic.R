library(dslabs)
data(murders)

# vector arithmetic operations apply element-wise
x <- c(1, 2, 3, 4)
y <- c(5, 6, 7, 8)

2.5 * x
x - mean(x)

x + y
x - y
x * y
x / y

# in sorting.R, we saw that California has the most murders
# however, it can easily be shown that it also has the most population
murders$state[which.max(murders$population)]

# the murder rate per capita is a better measure of safety
murder_rate = murders$total/murders$population * 100000

murder_order <- data.frame(murder_rate = murders$state[order(murder_rate, decreasing = TRUE)])
murder_order
