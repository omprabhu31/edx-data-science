# the data.table package is faster and better at handling & wrangling data than the tidyverse package
library(data.table)
library(tidyverse)
library(dplyr)
library(dslabs)
data(murders)

# functions in data.table require the data frame to first be converted to a data table
murders <- setDT(murders)

# there are 2 methods to selecting data using data.table
murders[, c("state", "region")] |> head()
murders[, .(state, region)] |> head()

# we used the following notation to add columns to dataframes in dplyr
murders <- mutate(murders, rate = total / population * 100000)

# mutating in data.table
murders[, rate := total / population * 100000]
head(murders)

# we can add multiple columns as shown
murders[, ":="(rate = total / population * 100000, rank = rank(population))]
head(murders)

# note that the := operator in data.table changes by reference
x = data.table(a = 1)
y <- x

x[, a := 2]
y

y[, a := 1]
x

# note that y is not separate from x - to avoid this, use copy() to make an actual copy
y <- copy(x)
x[,a := 2]
y
