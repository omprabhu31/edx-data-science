# we can encode a third variable in plots using either color hue or shape
# lets plot the measles rate in the US by year and state to see the effect that
# the introduction of the vaccine in 1963 had
library(tidyverse)
library(dslabs)
data(us_contagious_diseases)

# assign dat to the rate per 10,000 of measles, removing Alaska and Hawaii
# and adjusting for weeks reporting
dat <- us_contagious_diseases %>%
  filter(!state %in% c("Hawaii", "Alaska") & disease == "Measles") %>%
  mutate(rate = count / population * 10000 * 52/weeks_reporting) %>%
  mutate(state = reorder(state, rate, FUN=mean))

# plot disease rates per year in California
dat %>% filter(state == "California" & !is.na(rate)) %>%
  ggplot(aes(year, rate)) +
  geom_line() + ylab("Cases per 10,000") +
  geom_vline(xintercept = 1963, col = "blue")

# tile plot of disease rate by state and year
dat %>% ggplot(aes(year, state, fill = rate)) +
  geom_tile(color = "grey50") +
  scale_x_continuous(expand = c(0,0)) +
  scale_fill_gradientn(colors = RColorBrewer::brewer.pal(9, "Reds"), trans = "sqrt") +
  geom_vline(xintercept = 1963, col = "blue") +
  theme_minimal() + theme(panel.grid = element_blank()) +
  ggtitle("Measles") + xlab("") + ylab("")

# line plot of measles rate by state and year
avg <- us_contagious_diseases %>% filter(disease == "Measles") %>% group_by(year) %>%
  summarize(us_rate = sum(count, na.rm = TRUE)/sum(population, na.rm=TRUE)*10000)

# make line plot of measles rate by year by state
dat %>% filter(!is.na(rate)) %>% ggplot() +
  geom_line(aes(year, rate, group = state), color = "grey50",
            show.legend = FALSE, alpha= 0.2, size = 1) +
  geom_line(mapping = aes(year, us_rate), data = avg, size = 1, color = "black") +
  scale_y_continuous(trans = "sqrt", breaks = c(5, 25, 125, 300)) +
  ggtitle("Measles cases per 10,000 by state") +
  xlab("") + ylab("") +
  geom_text(data = data.frame(x = 1955, y = 50), mapping = aes(x, y, label = "US average"),
            color = "black") +
  geom_vline(xintercept = 1963, col = "blue")
