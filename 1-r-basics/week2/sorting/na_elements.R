library(dslabs)
data(na_example)

# checking the structure
str(na_example)

# if some elements are NA, computation of the average will yield NA
mean(na_example)

# use is.na() to create a logical vector that tells which entries are NA
na_entries <- is.na(na_example)
na_entries[1:20]

# the ! operator can be used to obtain the entries which are not NA
not_na <- !na_entries
not_na[1:20]

na_example[not_na]
mean(na_example[not_na])
