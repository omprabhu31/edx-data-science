# a distribution is a function that shows possible values of a variable
# and how often those values occur

library(dslabs)
data(heights)

# for categorical variables, the distribution described the proportion of each category
# for example, the distribution of the sex variable is given as below
freq_table <- table(heights$sex)
freq_table             # total number of males and females
prop.table(freq_table) # proportion of males and females

# for continuous data, a frequency table is not recommended because most values are unique
# let's see what happens if we create a freq table of all heights
prop.table(table(heights$height))

# for continuous data, the cumulative distribution function is a better measure
# the proportion of data below a value 'a' is represented as: F(a) = Pr(x <= a)
# the proportion of data between values 'a' & 'b' is then given by F(b) - F(a)

# the summary of continuous data is better represented using histograms or smooth
# density plots (essentially histograms with infinitesimally small bins)
