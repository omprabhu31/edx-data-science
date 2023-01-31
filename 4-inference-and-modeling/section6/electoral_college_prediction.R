library(tidyverse)
library(dslabs)
data("polls_us_election_2016")
head(results_us_election_2016)

results_us_election_2016 %>% arrange(desc(electoral_votes)) %>% top_n(5, electoral_votes)

# computing average and standard deviation for each state
results <- polls_us_election_2016 %>% 
  filter(state != "U.S." & !grepl("CD", state) & enddate >= "2016-10-31" &
           (grade %in% c("A+", "A", "A-", "B+") | is.na(grade))) %>%
  mutate(spread = (rawpoll_clinton - rawpoll_trump)/100) %>% group_by(state) %>%
  summarize(avg = mean(spread), sd = sd(spread), n = n()) %>%
  mutate(state = as.character(state))

# 10 closest races - battleground states
results %>% arrange(abs(avg))

# joining electoral college and votes
results <- left_join(results, results_us_election_2016, by = "state")

# filter out states with no polls
results_us_election_2016 %>% filter(!state %in% results$state)

# for states with just one poll, assign sd as median of the sd of other states
results <- results %>% mutate(sd = ifelse(is.na(sd), median(results$sd, na.rm = TRUE), sd))

# calculating posterior mean and standard error
mu <- 0
tau <- 0.02 # historical data

results %>% mutate(sigma = sd/sqrt(n),
                   B = sigma^2 / (sigma^2 + tau^2),
                   posterior_mean = B*mu + (1-B)*avg,
                   posterior_se = sqrt(1/(1/sigma^2 + 1/tau^2))) %>%
  arrange(abs(posterior_mean))

# monte carlo simulation of election night results without general bias
mu <- 0
tau <- 0.02
clinton_EV <- replicate(1000, {
  results %>% mutate(sigma = sd/sqrt(n),
                     B = sigma^2/ (sigma^2 + tau^2),
                     posterior_mean = B*mu + (1-B)*avg,
                     posterior_se = sqrt( 1 / (1/sigma^2 + 1/tau^2)),
                     simulated_result = rnorm(length(posterior_mean), posterior_mean, posterior_se),
                     clinton = ifelse(simulated_result > 0, electoral_votes, 0)) %>%
    summarize(clinton = sum(clinton)) %>%    # total electoral votes for Clinton
    .$clinton + 7    # 7 votes for Rhode Island and DC
})
mean(clinton_EV > 269) # 269 electoral votes required to win the election

# histogram of outcomes
data.frame(clinton_EV) %>%
  ggplot(aes(clinton_EV)) +
  geom_histogram(binwidth = 1) +
  geom_vline(xintercept = 269)

# monte carlo simulation of election night results with general bias
# assume general bias for each state = 0.03
mu <- 0
tau <- 0.02
sigma_b <- 0.03
clinton_EV2 <- replicate(1000, {
  results %>% mutate(sigma = sqrt(sd^2/n + sigma_b^2),
                     B = sigma^2/ (sigma^2 + tau^2),
                     posterior_mean = B*mu + (1-B)*avg,
                     posterior_se = sqrt( 1 / (1/sigma^2 + 1/tau^2)),
                     simulated_result = rnorm(length(posterior_mean), posterior_mean, posterior_se),
                     clinton = ifelse(simulated_result > 0, electoral_votes, 0)) %>%
    summarize(clinton = sum(clinton)) %>%    # total electoral votes for Clinton
    .$clinton + 7    # 7 votes for Rhode Island and DC
})
mean(clinton_EV2 > 269) # 269 electoral votes required to win the election

# histogram of outcomes
data.frame(clinton_EV2) %>%
  ggplot(aes(clinton_EV2)) +
  geom_histogram(binwidth = 1) +
  geom_vline(xintercept = 269)
