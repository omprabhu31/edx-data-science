# define a vector x consisting of male heights
library(tidyverse)
library(dslabs)
data(heights)

index <- heights$sex == "Male"
x <- heights$height[index]

# calculate the average and standard deviations
average <- mean(x) # sum(x) / length(x)
std_dev <- sd(x)   # sqrt(sum((x - average) ^ 2) / length(x))

# NOTE: the built-in sd() function computes the SD, but it divides by length(x) - 1
# for most data sets, however, this isn't a big deal

# to calculate standard units or the z-score
z <- scale(x)      # (x - average) / std_dev

# calculate the proportion of values within 2 SD
mean(abs(z < 2))
