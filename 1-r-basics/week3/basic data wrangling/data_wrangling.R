library(dplyr)
library(dslabs)
data(murders)

# we can add a column to the table using the mutate() function
murders <- mutate(murders, rate = total / population * 100000)
head(murders)

# we can filter rows based on a certain condition using filter()
filter(murders, rate <= 0.71)

# we can only choose to display certain variable columns using select()
new_table <- select(murders, state, region, rate)
filter(new_table, rate <= 0.71)

# the above operation of selecting and filtering can be compacted using the pipe %>%
# each instance of %>% 'pipes' the results of one function to another function
murders %>% select(state, region, rate) %>% filter(rate <= 0.71)

# if we want to display only state, population, total and rate for states in the northeast region
murders %>% filter(region == "Northeast") %>% select(state, population, total, rate)
