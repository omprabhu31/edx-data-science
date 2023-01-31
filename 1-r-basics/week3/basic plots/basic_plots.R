library(dslabs)
library(dplyr)
data(murders)

# we use plot() for a simple scatter plot
pop_in_mil <- murders$population / 1000000
total <- murders$total

plot(pop_in_mil, total)

# hist() is used to plot histograms
murder_rate <- murders$total / murders$population * 100000
hist(murder_rate)

# boxplot of murder rate by region
boxplot(murder_rate~region, data = murders)
