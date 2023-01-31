library(tidyverse)
library(dslabs)
data(polls_us_election_2016)
names(polls_us_election_2016)

# filter polls for US which occurred within one week of the election 
# and have either no grade or a grade higher than B
polls <- polls_us_election_2016 %>% filter(state == "U.S." & enddate >= "2016-10-31"
                                           & (grade %in% c('A+', 'A', 'A-', 'B+') | 
                                                is.na(grade)))

# add a field for spread in each poll
polls <- polls %>% mutate(spread = (rawpoll_clinton - rawpoll_trump)/100)

# calculate spread for aggregated data
d_hat <- polls %>% summarize(d_hat = sum(spread * samplesize)/sum(samplesize)) %>%
  pull(d_hat)
p_hat <- (d_hat+1)/2
margin_of_error <- 2 * 1.96 * sqrt(p_hat*(1-p_hat)/sum(polls$samplesize))

d_hat
margin_of_error

# however the actual spread was 2.1%, which is outside of the 95% confidence interval
# a histogram of the spread shows that the assumption of the data being normal is not valid
polls %>% ggplot(aes(spread)) + geom_histogram(color = 'black', binwidth = 0.01)

# this is because different pollsters have different number of polls
# furthermore, each pollster has a different bias, confirmed by the below calculation
polls %>% group_by(pollster) %>% filter(n() >= 6) %>%
  summarize(se = 2 * sqrt(p_hat*(1-p_hat)/median(samplesize)))

# plot showing spread and bias across different pollsters
polls %>% group_by(pollster) %>% filter(n() >= 6) %>% ggplot(aes(pollster, spread)) +
  geom_point() + theme(axis.text.x = element_text(angle = 90, hjust = 1))
