# coercion is an attempt by R to be flexible with data types
x <- c(1, "canada", 3)
x

# characters can sometimes be converted to numbers and vice-versa
a <- c(1, 2, 3)
b <- as.character(a)
b

c <- as.numeric(b)
c

# coercion may not always work, and display NA as an output
x <- c(1, "canada", 3)
as.numeric(x)
