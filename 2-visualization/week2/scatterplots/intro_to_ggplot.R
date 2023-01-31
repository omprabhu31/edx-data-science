library(tidyverse)
library(dslabs)
data(murders)

# STEP 1: define a ggplot object
p <- ggplot(data = murders)
# this can also be done in a more simple way: p <- ggplot(murders)
# another alternative way is: p <- murders %>% ggplot()

#-------------------------------------------------------------------------------

# STEP 2: add points layer to ggplot object
p + geom_point(aes(population / 10^6, total))

#-------------------------------------------------------------------------------

# STEP 3: add text layer to scatterplot
p + geom_point(aes(population / 10^6, total)) + 
  geom_text(aes(population / 10^6, total, label = abb))

# note that aes() is able to understand variables like population, total & abb
# not all ggplot functions can access variables that are not globally defined

#------------------------------(optional step)----------------------------------

# OPTIONAL STEP 1: modify arguments to geometry functions

# change the size of the points
p + geom_point(aes(population / 10^6, total), size = 3) + 
  geom_text(aes(population / 10^6, total, label = abb))

# move text labels slightly to the right
p + geom_point(aes(population / 10^6, total), size = 3) + 
  geom_text(aes(population / 10^6, total, label = abb), nudge_x = 1)

#------------------------------(optional step)----------------------------------

# OPTIONAL STEP 2: use global aesthetic mappings

# to avoid adding mappings multiple times (eg: in geom_point and geom_text),
# we can use global aesthetic mappings
p <- murders %>% ggplot(aes(population / 10^6, total, label = abb))
p + geom_point(size = 3) + geom_text(nudge_x = 1)

# local aesthetic mappings, if any, will override global mappings
p + geom_point(size = 3) + geom_text(aes(x = 10, y = 800, label = "Hello there!"))

#-------------------------------------------------------------------------------

# STEP 4: scale the axes to log10 values
# note that we also have to change the geom_text nudge value
p + geom_point(size = 3) + geom_text(nudge_x = 0.05) + 
  scale_x_continuous(trans = "log10") + scale_y_continuous(trans = "log10")

# an alternative way of doing this is as follows:
p + geom_point(size = 3) + geom_text(nudge_x = 0.05) + 
  scale_x_log10() + scale_y_log10()

#-------------------------------------------------------------------------------

# STEP 5: add labels and title
p + geom_point(size = 3) + geom_text(nudge_x = 0.05) + 
  scale_x_log10() + scale_y_log10() + 
  xlab("Population in millions (log scale)") + 
  ylab("Total number of murders (log scale)") + 
  ggtitle("US Gun Murders in 2010")

#-------------------------------------------------------------------------------

# STEP 6: change color of the points
# we first need to redefine p to be everything except the points layer
p <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) + 
  geom_text(nudge_x = 0.05) + scale_x_log10() + scale_y_log10() + 
  xlab("Population in millions (log scale)") + 
  ylab("Total number of murders (log scale)") + 
  ggtitle("US Gun Murders in 2010")

# make all points blue
p + geom_point(size = 3, color = "blue")

# color code points by region
p + geom_point(aes(col = region), size = 3)

#-------------------------------------------------------------------------------

# STEP 7: add line with average murder rate

# define average murder rate
r <- murders %>% summarize(rate = sum(total) / sum(population) * 10^6) %>% pull(rate)

# basic line with average murder rate for the country
p + geom_point(aes(col = region), size = 3) + 
  geom_abline(intercept = log10(r))

# change line to dashed & dark grey, line under points
p <- p + geom_abline(intercept = log10(r), lty = 2, color = "darkgrey") +
  geom_point(aes(col = region), size = 3)

#-------------------------------------------------------------------------------

# STEP 8: modify legend

p <- p + scale_color_discrete(name = "Region") # capitalize legend title
 
#-------------------------------------------------------------------------------

# STEP 9: add-on packages (ggthemes, ggrepel, etc)

# using ggthemes to change the theme
library(ggthemes)
p + theme_economist()

# using ggrepel to avoid labels overlaying on top of each other
library(ggrepel)
# we then use the geom_text_repel() function to separate the labels

#-------------------------------------------------------------------------------

# creating the entire plot from scratch

r <- murders %>% summarize(rate = sum(total) / sum(population) * 10^6) %>% pull(rate)

p <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) + # step 1
  geom_abline(intercept = log10(r), lty = 2, color = "darkgrey") +    # step 7
  geom_point(aes(col = region), size = 3) +                           # step 2 + 6
  geom_text_repel() +                                                 # step 3 + 9
  scale_x_log10() + scale_y_log10() +                                 # step 4
  xlab("Population in millions (log scale)") +                        # step 5
  ylab("Total number of murders (log scale)") +                       # step 5
  ggtitle("US Gun Murders in 2010") +                                 # step 5
  scale_color_discrete(name = "Region") +                             # step 8
  theme_economist()                                                   # step 9
