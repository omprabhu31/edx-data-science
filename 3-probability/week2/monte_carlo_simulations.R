library(tidyverse)
library(dslabs)
data(heights)
x <- heights %>% filter(sex=="Male") %>% pull(height)
set.seed(1)

# the rnorm() function generates n random numbers from the normal distribution
# with a specified average and standard deviation
n <- length(x)
avg <- mean(x)
s <- sd(x)
simulated_heights <- rnorm(n, avg, s)

# plot distribution of simulated heights
data.frame(simulated_heights = simulated_heights) %>% ggplot(aes(simulated_heights)) +
  geom_histogram(color = "black", binwidth = 2)

# monte carlo simulation of tallest person over 7 feet
B <- 800
tallest <- replicate(B, {
  simulated_data <- rnorm(B, avg, s)
  max(simulated_data)
})
mean(tallest >= 7*12)
