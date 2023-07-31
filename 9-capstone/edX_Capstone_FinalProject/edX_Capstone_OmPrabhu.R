###############################################################################
# Section 1: Importing packages and loading the dataset

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")

library(caret)
library(kernlab)

# attach the iris dataset to the environment
data(iris)

# rename the dataset
dataset <- iris

###############################################################################
# Section 2: Create a train-test split

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

###############################################################################
# Section 3: Analyse the data

# first 6 rows of the data
head(dataset)

# dimensions of dataset
dim(dataset)

# types for each attribute
sapply(dataset, class)

# levels for the class
levels(dataset$Species)

# summarize the class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

# summarize the attribute
summary(dataset)

###############################################################################
# Section 4: Exploratory data analysis

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

# boxplot for each attribute
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

# box plots for each attribute separated by class labels
featurePlot(x=x, y=y, plot="box")

###############################################################################
# Section 5: Build and evaluate models

# set up 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# LDA
set.seed(42)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

# Classification Tree
set.seed(42)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

# kNN
set.seed(42)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

# SVM
set.seed(42)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)

# Random Forest
set.seed(42)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, ct=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize best model
print(fit.lda)

###############################################################################
# Section 6: Use the best model for validation

# accuracy of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)
