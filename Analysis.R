# Load library and packages

library(caret)

# Download files

#download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv", method = "curl")
#download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv", method = "curl")

pml.training <- read.csv("pml-training.csv")
pml.testing <- read.csv("pml-testing.csv")

names(pml.training)
head(pml.training)

# Remove columns containing NAs
pml.training2 <- pml.training[, colSums(is.na(pml.training)) == 0]
pml.testing2 <- pml.testing[, colSums(is.na(pml.testing)) == 0]

# Additionally, it's worth noting that some of the variables in the data set do not come from accelerometer measurements and record experimental setup or participants' data. Treating those as potential confounders is a sane thing to do, so in addition to predictors with missing data, I also discarded the following variables: X, user_name, raw_timestamp_part1, raw_timestamp_part2, cvtd_timestamp, new_window and num_window.
pml.training3 <- pml.training2[, !grepl("X|user_name|timestamp|window", colnames(pml.training2))]
pml.testing3 <- pml.testing2[, !grepl("X|user_name|timestamp|window", colnames(pml.testing2))]

# The datatset was originally used by the authors to characterize the performance classes by identifying the most relevant features. The data comes with the column “new_window” identifying a pre-determined time window to calculate the features of the distributions of the measurements. When “yes”, the features fill the columns of max/min value, averag, skewness, etc. Since the test set has only 20 observations to predict with, these columns can not be calculated in the testing phase and so will be dropped in the training phase.

pml.training4 <- pml.training3[, !grepl("^max|^min|^ampl|^var|^avg|^stdd|^ske|^kurt", colnames(pml.training3))]
pml.testing4 <- pml.testing3[, !grepl("^max|^min|^ampl|^var|^avg|^stdd|^ske|^kurt", colnames(pml.testing3))]

# Cross validation (70% training and 30% validation)
set.seed(23222)
#folds <- createFolds(y = pml.training3$classe, k = 10, list = TRUE, returnTrain = TRUE)
inTrain <- createDataPartition(y = pml.training4$classe, p = 0.7, list = FALSE)
pml.train <- pml.training4[inTrain, ]
pml.valid <- pml.training4[-inTrain, ]
pml.testing <- pml.testing4

# Plots

pml.corr <- cor(pml.train[, -53])
heatmap(pml.corr)
library(corrplot)
corrplot(pml.corr, method = "color")

# Most predictors do not exhibit a high degree of correlation, however some variables are highly correlated.

M <- abs(pml.corr)
diag(M) <- 0
high.corr <- which(M > 0.8, arr.ind = TRUE)
for (i in 1:nrow(high.corr)) {
    print(names(pml.train)[high.corr[i, ]])
}

# Solution : use PCA to pick the combination of predictors that captures the most information possible (benefits : reduced number of predictors and reduced noise).

preProc.pca <- preProcess(pml.train[, -53], method  = "pca", thresh = 0.95) # 0.9
pml.train.pca <- predict(preProc.pca, pml.train[, -53])
pml.valid.pca <- predict(preProc.pca, pml.valid[, -53])
pml.testing.pca <- predict(preProc.pca, pml.testing[, -53])
print(preProc.pca)

# Use of random forests We chose to specify the use of a cross validation method when applying the random forest routine in the 'trainControl()' parameter. Without specifying this, the default method (bootstrapping) would have been used. The bootstrapping method seemed to take a lot longer to complete, while essentially producing the same level of 'accuracy'.

modFit <- train(pml.train$classe ~ ., method = "rf", data = pml.train.pca, trControl = trainControl(method = "cv", 5)) # 5
modFit

# We now review the relative importance of the resulting principal components of the trained model, 'modelFit'.

varImpPlot(modFit$finalModel, sort = TRUE)

# Cross validation

pml.pred.valid <- predict(modFit, pml.valid.pca)
confusionMatrix(pml.valid$classe, pml.pred.valid)

# Out of sample error

OoSE <- 1 - as.numeric(confusionMatrix(pml.valid$classe, pml.pred.valid)$overall[1])

# Performance on test dataset (Correct values : "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B")

pml.pred.test <- predict(modFit, pml.testing.pca)

correct <- c("B", "A", "B", "A", "A", "E", "D", "B", "A", "A", "B", "C", "B", "A", "E", "E", "A", "B", "B", "B")
correct <- factor(correct)
correct == pml.pred.test