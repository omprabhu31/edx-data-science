# the function c() is useful for creating vectors
# c stands for concatenate
codes <- c(380, 124, 818, 238)
codes

country <- c("italy", "canada", "egypt", "india")
country

# we use the names() function to name the elements of a vector
# here, the elements in country label the elements in the codes vector
names(codes) <- country
codes
class(codes)

# we can also name the elements of a numeric vector
# both the representations below are equivalent
codes <- c(italy = 380, canada = 124, egypt = 818, india = 238)
codes

codes <- c("italy" = 380, "canada" = 124, "egypt" = 818, "india" = 238)
codes
class(codes)

# another function that can be used to create vectors is seq()
seq(1,10) # -> integer class
seq(1,10,3) # overrides the default sequence gap -> numeric class
1:10 # shorthand code for integer sequences -> integer class

# use [] brackets to access particular subsets of a vector
codes[2]
codes[c(2,4)]
codes[seq(1,4,3)]
codes[1:3]

# if the entries of a vector are named, they can be accessed
# by referring to their name
codes["canada"]
codes[c("italy", "egypt", "india")]
