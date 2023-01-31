library(tidyverse)
library(dplyr)
library(dslabs)
data(murders)

murders <- mutate(murders, rate = total / population * 10^5)

# just like sort() and order() with vectors, the function arrange() can be used to sort dataframes
murders %>% arrange(population) %>% head()
murders %>% arrange(desc(population)) %>% head()

# we can use multiple coloums to break ties in the previous columns
murders %>% arrange(region, rate) %>% head()

# note how head() only displays the first 6 entries
# we can use the top_n() function to display the top n entries
murders %>% top_n(10, population)

# however, top_n() does not sort the entries
murders %>% arrange(desc(population)) %>% top_n(10, population)
