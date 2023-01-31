library(tidyverse)
library(dslabs)
data(murders)

p <- murders %>% ggplot(aes(population / 10^6, total, label = abb))

p1 <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) +
  geom_point(size = 3)

p2 <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) +
  geom_point(size = 3) +                           
  geom_text(nudge_x = 1)   

p3 <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) +
  geom_point(size = 3) +                           
  geom_text(nudge_x = 0.05) +                                                 
  scale_x_log10() + scale_y_log10()

p4 <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) +
  geom_point(size = 3) +                           
  geom_text(nudge_x = 0.05) +                                                 
  scale_x_log10() + scale_y_log10() +                         
  xlab("Population in millions (log scale)") +                 
  ylab("Total number of murders (log scale)") +                  
  ggtitle("US Gun Murders in 2010")

p5 <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) +
  geom_point(aes(col = region), size = 3) +                           
  geom_text(nudge_x = 0.05) +                                                 
  scale_x_log10() + scale_y_log10() +                         
  xlab("Population in millions (log scale)") +                 
  ylab("Total number of murders (log scale)") +                  
  ggtitle("US Gun Murders in 2010")

p6 <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) +
  geom_abline(intercept = log10(r), lty = 2, color = "darkgrey") +    
  geom_point(aes(col = region), size = 3) +                           
  geom_text(nudge_x = 0.05) +                                                 
  scale_x_log10() + scale_y_log10() +                         
  xlab("Population in millions (log scale)") +                 
  ylab("Total number of murders (log scale)") +                  
  ggtitle("US Gun Murders in 2010")

p7 <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) +
  geom_abline(intercept = log10(r), lty = 2, color = "darkgrey") +    
  geom_point(aes(col = region), size = 3) +                           
  geom_text(nudge_x = 0.05) +                                                 
  scale_x_log10() + scale_y_log10() +                         
  xlab("Population in millions (log scale)") +                 
  ylab("Total number of murders (log scale)") +                  
  ggtitle("US Gun Murders in 2010") +                                
  scale_color_discrete(name = "Region")

library(ggthemes)
library(ggrepel)

p8 <- murders %>% ggplot(aes(population / 10^6, total, label = abb)) +
  geom_abline(intercept = log10(r), lty = 2, color = "darkgrey") +    
  geom_point(aes(col = region), size = 3) +                           
  geom_text_repel() +                                                 
  scale_x_log10() + scale_y_log10() +                         
  xlab("Population in millions (log scale)") +                 
  ylab("Total number of murders (log scale)") +                  
  ggtitle("US Gun Murders in 2010") +                                
  scale_color_discrete(name = "Region") +                             
  theme_economist()  

library(gridExtra)
grid.arrange(p, p1, p2, p3, p4, p5, p6, p7, p8, ncol = 3)
