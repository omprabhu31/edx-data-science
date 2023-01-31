# we can create data frames using the data.frame() function
# however, by default, all characters are converted to factors
grades <- data.frame(names = c("Sam", "Nick", "Joe", "Tom"),
                     exam_1 = c(90, 75, 80, 95),
                     exam_2 = c(60, 85, 90, 70))
class(grades$names)

# to avoid this, we use the stringsAsFactors argument
# this is NOT necessary as of R 4.0 and above
grades <- data.frame(names = c("Sam", "Nick", "Joe", "Tom"),
                     exam_1 = c(90, 75, 80, 95),
                     exam_2 = c(60, 85, 90, 70),
                     stringsAsFactors = FALSE)
class(grades$names)
