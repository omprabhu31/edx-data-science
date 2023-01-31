# Bayes' theorem -> Pr(A | B) = Pr(B | A) * Pr(A) / Pr(B)
# consider a cystic fibrosis test with a test accuracy of 99%
# i.e. 99% test accuracy when the disease is present -> Pr(+ | D=1) = 0.99
# i.e. 99% test accuracy when the disease is absent -> Pr(- | D=0) = 0.99

# thus, given that average prevalence of cystic fibrosis is about 1 in 3900 (~0.00025) 
# we can calculate the probability that a person diagnosed positive actually has the disease

# Pr(D=1 | +) = Pr(+ | D=1) * Pr(D=1) / Pr(+)
#             = Pr(+ | D=1) * Pr(D=1) / [{Pr(+ | D=1) * Pr(D=1)} + {Pr(+ | D=0) * Pr(D=0)}]
#             = 0.99 * 0.00025 / [{0.99 * 0.00025} + {0.01 * 0.99975}]
#             = 0.02

# monte carlo simulation to verify the results
N <- 100000
p <- 0.00025

X <- sample(c("Disease", "Healthy"), N, replace = TRUE, prob = c(p, 1-p))
N_D <- sum(X == "Disease")
N_H <- sum(X == "Healthy")

# for each person determine whether the test is + or -
accuracy <- 0.99
test <- vector("character", N)
test[X == "Disease"] <- sample(c("+", "-"), N_D, replace = TRUE, 
                               prob = c(accuracy, 1-accuracy))
test[X == "Healthy"] <- sample(c("+", "-"), N_H, replace = TRUE, 
                               prob = c(1-accuracy, accuracy))

table(X, test)
