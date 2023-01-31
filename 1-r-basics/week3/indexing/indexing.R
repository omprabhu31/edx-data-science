library(dslabs)
data(murders)

murder_rate <- murders$total / murders$population * 100000

# we can use logicals to index vectors
# for example, we move to the US from italy (murder rate 0.71)
# we want to go to a state with a lower or equal rate
index <- murder_rate <= 0.71
murders$state[index] # safe states
sum(index) # number of safe states

# suppose we want to go to a state in west US and having murder rate less than 1
safe <- murder_rate <= 1
west <- murders$region == "West"

index <- safe & west
murders$state[index]

# suppose we avoid states with more than 10000000 population or with murder rate higher than 3
unsafe <- murder_rate >= 3
overpop <- murders$population >= 10000000

index <- unsafe | overpop
murders$state[index]
