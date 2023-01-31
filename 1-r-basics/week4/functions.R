# the R function called function() is used to define a new function
amgm <- function(x, y, arithmetic = TRUE) {
  ifelse(arithmetic, (x + y) / 2, (x * y)^(1/2))
}

amgm(3, 5)
amgm(3, 5, FALSE)
