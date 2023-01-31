library(data.table)
library(tidyverse)
library(dplyr)
library(dslabs)
data(murders)

murders <- setDT(murders)
murders[, ":="(rate = total / population * 100000, rank = rank(-population))]

# sorting data in data.table
murders[order(population)] |> head()
murders[order(population, decreasing = TRUE)] |> head()

# sorting by more than one variable
murders[order(region, rate)] |> head()
