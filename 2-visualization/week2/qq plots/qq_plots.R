library(tidyverse)
library(dslabs)
data(heights)

p <- heights %>% filter(sex == "Male") %>% ggplot(aes(sample = height))

# basic qq plot, plotted against the standard normal distribution
p + geom_qq()

# qq plot, plotted against mean & sd of our data
params <- heights %>% filter(sex == "Male") %>% 
  summarize(mean = mean(height), sd = sd(height))

# plotting against custom mean & sd, and comparing with normal distribution
p + geom_qq(dparams = params) + geom_abline() +
  xlab("Male height in inches") + ggtitle("QQ plot of male heights")

# qq plot of scaled data against the standard normal distribution
p <- heights %>% filter(sex == "Male") %>% ggplot(aes(sample = scale(height))) +
  geom_qq() + geom_abline()
