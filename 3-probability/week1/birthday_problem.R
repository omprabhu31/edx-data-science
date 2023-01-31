# checking for duplicated birthdays in a class of 50 students
students <- 50
bdays <- sample(1:365, students, replace = TRUE)
set.seed(2022 - 9 - 29)

n <- 10000
results <- replicate(n, {
  bdays <- sample(1:365, students, replace = TRUE)
  any(duplicated(bdays))
})
mean(results)
