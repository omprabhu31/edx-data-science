# loading the dslabs package and murders dataset
library(dslabs)
data("murders") # can also be written without parentheses

# data frames can be thought of as tables with rows representing 
# observations and columns representing different variables

# determining that murders dataset is of the data frame class
class(murders)
# finding out more about the structure of the object
str(murders)

# display the first 6 lines of the data frame
head(murders)
# display variable names
names(murders)

# use the accessor $ to access all values in a particular column
# output is a vector (vector class dependent on variable class)
pop <- murders$population
pop
length(pop)
class(pop)

states <- murders$state
states
class(states)

# factor is a data type that stores categorical data
reg <- murders$region
class(reg)
# obtain the levels of a factor
levels(reg)

# logical vectors are either true or false
z <- 3==2
z
class(z)

