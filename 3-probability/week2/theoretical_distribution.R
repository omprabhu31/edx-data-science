library(tidyverse)
library(dslabs)
data(heights)
x <- heights %>% filter(sex == "Male") %>% .$height

# pnorm() gives the value of the CDF for the normal distribution
1 - pnorm(70.5, mean(x), sd(x))

# plot distribution of exact heights in the data set
plot(prop.table(table(x)), xlab = "a = Heights in inches", ylab = "Pr(x = a)")

# clearly many students have rounded off their heights (discretization)
# thus CDFs are useful for categorical data, not discrete data
# probabilities in actual data over length 1 containing integer values
mean(x <= 68.5) - mean(x <= 67.5)
mean(x <= 69.5) - mean(x <= 68.5)
mean(x <= 70.5) - mean(x <= 69.5)

# probabilities in normal approximation match well
pnorm(68.5, mean(x), sd(x)) - pnorm(67.5, mean(x), sd(x))
pnorm(69.5, mean(x), sd(x)) - pnorm(68.5, mean(x), sd(x))
pnorm(70.5, mean(x), sd(x)) - pnorm(69.5, mean(x), sd(x))

# probabilities in intervals without integer values do not match well
mean(x <= 70.9) - mean(x <= 70.1)
pnorm(70.9, mean(x), sd(x)) - pnorm(70.1, mean(x), sd(x))
