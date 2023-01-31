library(dslabs)
library(dplyr)
data(heights)

# the qth quantile is the value at which q% of the observations are less than or
# equal to that value
quantile(heights$height, 0.6)

# percentiles are the quantiles that divide a dataset into 100 intervals each 
# with 1% probability
p <- seq(0.01, 0.99, 0.01)
percentiles <- quantile(heights$height, p)
percentiles

# quartiles divide a set into 4 intervals with 25% probability each
summary(heights$height)

percentiles[names(percentiles) == "25%"] # 1st quartile
percentiles[names(percentiles) == "75%"] # 3rd quartile

# the qnorm(p, mu, sigma) function gives the theoretical value of a quantile 
# with probability p of observing a value less than or equal to the quantile value
# given a normal distribution with mean mu and standard deviation sigma
male_heights <- heights %>% filter(sex == "Male")
theoretical_quantiles <- qnorm(p, mean(male_heights$height), sd(male_heights$height))

# qnorm() and pnorm() are inverse functions
pnorm(qnorm(0.025))
