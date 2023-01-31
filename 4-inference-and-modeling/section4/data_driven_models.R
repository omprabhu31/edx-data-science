library(tidyverse)
library(dslabs)
data(polls_us_election_2016)
names(polls_us_election_2016)

# filter polls for US which occurred within one week of the election 
# and have either no grade or a grade higher than B
polls <- polls_us_election_2016 %>% filter(state == "U.S." & enddate >= "2016-10-31"
                                           & (grade %in% c('A+', 'A', 'A-', 'B+') | 
                                                is.na(grade)))
polls <- polls %>% mutate(spread = (rawpoll_clinton - rawpoll_trump)/100)

# filter the last-reported result before the election
one_poll_per_pollster <- polls %>% group_by(pollster) %>% 
  filter(enddate == max(enddate)) %>% ungroup()

# plot of spread estimates
one_poll_per_pollster %>% ggplot(aes(spread)) + 
  geom_histogram(color = 'black', binwidth = 0.01)

# instead of using an urn model with 0s and 1s, we use a different urn model that contains
# poll results from all possible pollsters

# the central limit theorem still works to estimate the sample average of the random draws,
# since they are independent events
confidence_interval_95 <- one_poll_per_pollster %>% 
  summarize(avg = mean(spread), se = sd(spread)/sqrt(length(spread))) %>%
  mutate(start = avg - abs(qnorm(0.975))*se, end = avg + abs(qnorm(0.975))*se)
round(confidence_interval_95 * 100, 1) # still does not include zero, but is wide enough
                                       # to account for pollster bias
