library(tidyverse)
library(dslabs)
data(heights)

p <- heights %>% filter(sex == "Male") %>% ggplot(aes(x = height))

p + geom_density(fill = "orange", outline.type = "both") +
  xlab("Male heights in inches") +
  ggtitle("Smooth density plot of male heights")