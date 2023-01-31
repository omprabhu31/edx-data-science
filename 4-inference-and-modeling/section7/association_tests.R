# association tests can be used to determine the probability that a particular observation
# is due to random variability given categorical, binary or ordinal data

# load and inspect research funding rates object
library(tidyverse)
library(dslabs)
data(research_funding_rates)
research_funding_rates

# compute totals that were successful or not successful
totals <- research_funding_rates %>% select(-discipline) %>% # select all except discipline
  summarize_all(funs(sum)) %>% # calculate sum of all individual columns
  summarize(yes_men = awards_men,
            no_men = applications_men - awards_men,
            yes_women = awards_women,
            no_women =applications_women - awards_women)

# compare percentage of men/women with awards
totals %>% summarize(percent_men = yes_men / (yes_men + no_men) * 100,
                     percent_women = yes_women / (yes_women + no_women) * 100)

# fisher's exact test determines the probability of observing an outcome as extreme or 
# more extreme than the observed outcome given the null distribution

# fisher's colleague claimed that she could tell if milk was added before or after tea
# he formed 4 randomly ordered pairs of cups, each with one where milk was added before 
# and the other where milk was added after tea

# the null hypothesis is that she has no special powers and that she's simply guessing
# fisher derived the distribution assuming that her choices were random & independent

# as an example, suppose she picked 3 out of 4 correctly. do we believe she has a special
# ability based on this? thus, the actual question we're asking is:
# if the tester is guessing, what are the chances that she gets 3 or more correct?

# two-by-two table for the 'Lady Tasting Tea' problem
tab <- matrix(c(3,1,1,3), 2, 2)
rownames(tab) <- c("Poured Before", "Poured After")
colnames(tab) <- c("Guessed Before", "Guessed After")
tab

# p-value calculation with Fisher's Exact Test
fisher.test(tab, alternative = "greater")

# p.s. in the actual test, all of her choices were correct, thus actual p-value = 1/70
