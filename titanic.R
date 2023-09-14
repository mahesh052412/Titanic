#set the working directory
setwd("C:/Users/HP/OneDrive/Desktop/Kaggle/titanic/dataset")

#libraries used
library(caret)
library(ggplot2)
library(klaR)
library(dials)
library(rsample)
library(dplyr)
library(kernlab)
library(lattice)
library(e1071)
library(yardstick)

#read the data file
data_file <- read.csv("train.csv")

#Converting string values to numeric
data_file$Sex <- factor(data_file$Sex,
                        levels = c("male","female"),
                        labels = c(1,0))

data_file$Embarked <- factor(data_file$Embarked,
                             levels = c("C","Q","S"),
                             labels = c("1","2","3"))

#Assigning the missing values
data_file$Age[is.na(data_file$Age)] <- median(data_file$Age, na.rm = T)
data_file$Embarked[is.na(data_file$Embarked)] <- median(as.numeric(data_file$Embarked), na.rm = T)

#categorical data
ggplot(data_file, aes(x = Age)) +
  geom_histogram(bins = 100, binwidth = 1,
                 fill = "lightblue", 
                 color = "blue") +
  scale_x_continuous(breaks = c(0,10,20,30,40,50,60,70,80)) +
  theme_classic()

  

#K-Fold Cross-validation for the dataset
#setting seed to generate a random sampling 
set.seed(125)


#Split the dataset 
split_data <- initial_split(data_file[, -c(4, 9, 11)], prop = 0.8, strata = Survived)

#Dataset for training model
training_data <- split_data %>% training()

#Dataset for testing model
testing_data <- split_data %>% testing()

#Using Support vector machine for classification 
svm_clf <- svm(as.factor(Survived)~., training_data, kernel = "linear",
               cost = 10, scale = T)

#summary for SVM
summary(svm_clf)

#Predicting for testing dataset
y_pred <- predict(svm_clf, testing_data, type = 'class')

#Accuarcy from model for testing dataset
accuracy_vec(as.factor(testing_data$Survived),y_pred)

#Taking the test data
testcsv_data <- read.csv("test.csv") 

#converting string values to numeric
testcsv_data$Sex <- factor(testcsv_data$Sex,
                           levels = c("male", "female"),
                           labels = c(1,0))

testcsv_data$Embarked <- factor(testcsv_data$Embarked,
                                levels = c("C","Q","S"),
                                labels = c("1","2","3"))

#assiging the missing values for test data
testcsv_data$Age[is.na(testcsv_data$Age)] <- median(testcsv_data$Age, na.rm = T)
testcsv_data$Embarked[is.na(testcsv_data$Embarked)] <- median(as.numeric(testcsv_data$Embarked), na.rm = T)
testcsv_data$Fare[is.na(testcsv_data$Fare)] <- median(testcsv_data$Fare, na.rm = T)

#predicting for the test dataset
test_pred <- predict(svm_clf, testcsv_data, type = 'class')

#creatinf a new df
Outcome_values <- data.frame(testcsv_data$PassengerId, test_pred)
names(Outcome_values) <- c("PassengerId","Survived")
head(Outcome_values)

#converting dataframe to csv
write.csv(x = Outcome_values, file = "mysubmission.csv", row.names = FALSE)
