install.packages("caret")
install.packages("smotefamily")
library(caret)
library(corrplot)
library(smotefamily)

## Task 1: Import the dataset

creditcard <- read.csv("C:/Users/moharahman/Downloads/creditcardFraud.csv", header=TRUE)
View(creditcard)

### Task 2: Explore The Data

str(creditcard)
attach(creditcard)
creditcard$class <- as.factor(creditcard$class) #class as a factor variable

head(creditcard)
summary(creditcard) #checking missing values

summary(creditcard$class) #number of fraudulent and normal transaction in the original data

prop.table(summary(creditcard$class)) # % of fraudulent and normal transactions

hist(creditcard[,1:30]) #supposed to work, but not working 
corrplot(cor(creditcard[,1:30]))

### Task 3: Split the Data into Train and Test Sets

set.seed(1337)
train <- createDataPartition(creditcard$class,
                             p=.70,
                             times = 1,
                             list = F) #Splitting the data into training and testing dataset

train_data <- creditcard[train,] #original data
test_data <- creditcard[-train,] #original data

dim(train_data)/dim(creditcard) #Checking the proportion of observations allocated to each group
dim(test_data)/dim(creditcard)

prop.table(table(train_data$class)) #Class balance for training dataset
prop.table(table(test_data$class)) #Class balance for testing dataset

### Task 4: Compile Synthetically Balanced Training Datsets

train_smote <- SMOTE(train_data[,-31],train_data[,31], K=5) #SMOTE balacing technique
summary(train_smote)
t_smote <- train_smote$data
t_smote$class <- as.factor(t_smote$class)

train_adas <- ADAS(train_data[,-31],train_data[,31], K=5) #ADAS balacing technique
summary(train_adas)
t_adas <- train_adas$data
t_adas$class <- as.factor(t_adas$class)

train_dbsmote <- DBSMOTE(train_data[,-31],train_data[,31]) #DMSMOTE balacing technique
summary(train_dbsmote)
t_dbsmote <- train_dbsmote$data
t_dbsmote$class <- as.factor(t_dbsmote$class)

#Let's evaluate the Class distributions for Synthetic datasets

prop.table(table(t_smote$class))
prop.table(table(t_adas$class))
prop.table(table(t_dbsmote$class))

## Task 5: Original Data: Train Decision Tree, Naive Bayes, and LDA Models

ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     classProbs = TRUE, 
                     summaryFunction = twoClassSummary) #Global options that we will use across all of our trained models#

dt_orig <- train(class~.,
                 data=train_data,
                 method="rpart",
                 trControl=ctrl,
                   metric="ROC") # decision tree based on original data

nb_orig <- train(class~.,
                 data=train_data,
                 method="naive_bayes",
                 trControl=ctrl,
                 metric="ROC") # Naive Bayes based on original data


lda_orig <- train(class~.,
                 data=train_data,
                 method="lda",
                 trControl=ctrl,
                 metric="ROC") # Linear Discriminant Analysis based on original data

#Next, we will use the models we have trained using 
#the original imbalanced training dataset to generate predictions on the test dataset.

##decision tree model - Trained on original dataset

dt_orig_prediction <- predict(dt_orig, test_data, type = "prob") #step 1: dt model prediction

dt_orig_test <- factor(ifelse(dt_orig_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_dtorig <- posPredValue(dt_orig_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_dtorig <- sensitivity(dt_orig_test, test_data$class, positive = 'yes')
F1_dtorig <- (2*precision_dtorig*recall_dtorig)/(precision_dtorig+recall_dtorig)


##Naive Bayes Model - Trained on original dataset

nb_orig_prediction <- predict(nb_orig, test_data, type = "prob") #step 1: naive bayes model prediction

nb_orig_test <- factor(ifelse(nb_orig_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_nborig <- posPredValue(nb_orig_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_nborig <- sensitivity(nb_orig_test, test_data$class, positive = 'yes')
F1_nborig <- (2*precision_nborig*recall_nborig)/(precision_nborig+recall_nborig)

##LDA Model - Trained on original dataset

lda_orig_prediction <- predict(lda_orig, test_data, type = "prob") #step 1: naive bayes model prediction

lda_orig_test <- factor(ifelse(lda_orig_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_ldaorig <- posPredValue(lda_orig_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_ldaorig <- sensitivity(lda_orig_test, test_data$class, positive = 'yes')
F1_ldaorig <- (2*precision_ldaorig*recall_ldaorig)/(precision_ldaorig+recall_ldaorig)

### Task 6: SMOTE Balanced Data: Train Decision Tree, Naive Bayes, and LDA Models

#using the same code from step five, except this time with smote balance data

dt_smote <- train(class~.,
                 data=t_smote,
                 method="rpart",
                 trControl=ctrl,
                 metric="ROC") # decision tree based on SMOTE balanced data

nb_smote <- train(class~.,
                 data=t_smote,
                 method="naive_bayes",
                 trControl=ctrl,
                 metric="ROC") # Naive Bayes based on SMOTE balanced data


lda_smote <- train(class~.,
                  data=t_smote,
                  method="lda",
                  trControl=ctrl,
                  metric="ROC") # Linear Discriminant Analysis based on SMOTE balanced data

#Next, we will use the models we have trained using 
#SMOTE balanced data to generate predictions on the test dataset.

##decision tree model - Trained on SMOTE balanced data

dt_smote_prediction <- predict(dt_smote, test_data, type = "prob") #step 1: dt model prediction

dt_smote_test <- factor(ifelse(dt_smote_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_dtsmote <- posPredValue(dt_smote_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_dtsmote <- sensitivity(dt_smote_test, test_data$class, positive = 'yes')
F1_dtsmote <- (2*precision_dtsmote*recall_dtsmote)/(precision_dtsmote+recall_dtsmote)


##Naive Bayes Model - Trained on SMOTE balanced data

nb_smote_prediction <- predict(nb_smote, test_data, type = "prob") #step 1: naive bayes model prediction

nb_smote_test <- factor(ifelse(nb_smote_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_nbsmote <- posPredValue(nb_smote_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_nbsmote <- sensitivity(nb_smote_test, test_data$class, positive = 'yes')
F1_nbsmote <- (2*precision_nbsmote*recall_nbsmote)/(precision_nbsmote+recall_nbsmote)

##LDA Model - Trained on SMOTE balanced data

lda_smote_prediction <- predict(lda_smote, test_data, type = "prob") #step 1: naive bayes model prediction

lda_smote_test <- factor(ifelse(lda_smote_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_ldasmote <- posPredValue(lda_smote_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_ldasmote <- sensitivity(lda_smote_test, test_data$class, positive = 'yes')
F1_ldasmote <- (2*precision_ldasmote*recall_ldasmote)/(precision_ldasmote+recall_ldasmote)


### Task 7: ADASYN Balanced Data: Train Decision Tree, Naive Bayes, and LDA Models

#using the same code from step five/six, except this time with ADASYN balance data

dt_adas <- train(class~.,
                  data=t_adas,
                  method="rpart",
                  trControl=ctrl,
                  metric="ROC") # decision tree based on ADASYN Balanced Data

nb_adas <- train(class~.,
                  data=t_adas,
                  method="naive_bayes",
                  trControl=ctrl,
                  metric="ROC") # Naive Bayes based on ADASYN Balanced Data


lda_adas <- train(class~.,
                   data=t_adas,
                   method="lda",
                   trControl=ctrl,
                   metric="ROC") # Linear Discriminant Analysis based on ADASYN Balanced Data

#Next, we will use the models we have trained using 
#ADASYN Balanced Data to generate predictions on the test dataset.

##decision tree model - Trained on ADASYN Balanced Data

dt_adas_prediction <- predict(dt_adas, test_data, type = "prob") #step 1: dt model prediction

dt_adas_test <- factor(ifelse(dt_adas_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_dtadas <- posPredValue(dt_adas_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_dtadas <- sensitivity(dt_adas_test, test_data$class, positive = 'yes')
F1_dtadas <- (2*precision_dtadas*recall_dtadas)/(precision_dtadas+recall_dtadas)


##Naive Bayes Model - Trained on ADASYN Balanced Data

nb_adas_prediction <- predict(nb_adas, test_data, type = "prob") #step 1: naive bayes model prediction

nb_adas_test <- factor(ifelse(nb_adas_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_nbadas <- posPredValue(nb_adas_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_nbadas <- sensitivity(nb_adas_test, test_data$class, positive = 'yes')
F1_nbadas <- (2*precision_nbadas*recall_nbadas)/(precision_nbadas+recall_nbadas)

##LDA Model - Trained on ADASYN Balanced Data

lda_adas_prediction <- predict(lda_adas, test_data, type = "prob") #step 1: naive bayes model prediction

lda_adas_test <- factor(ifelse(lda_adas_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_ldaadas <- posPredValue(lda_adas_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_ldaadas <- sensitivity(lda_adas_test, test_data$class, positive = 'yes')
F1_ldaadas <- (2*precision_ldaadas*recall_ldaadas)/(precision_ldaadas+recall_ldaadas)


### Task 8: DB-SMOTE Balanced Data: Train Decision Tree, Naive Bayes, and LDA Models

#using the same code from step five/six, except this time with DB-SMOTE Balanced Data

dt_dbsmote <- train(class~.,
                 data=t_dbsmote,
                 method="rpart",
                 trControl=ctrl,
                 metric="ROC") # decision tree based on DB-SMOTE Balanced Data

nb_dbsmote <- train(class~.,
                 data=t_dbsmote,
                 method="naive_bayes",
                 trControl=ctrl,
                 metric="ROC") # Naive Bayes based on DB-SMOTE Balanced Data


lda_dbsmote <- train(class~.,
                  data=t_dbsmote,
                  method="lda",
                  trControl=ctrl,
                  metric="ROC") # Linear Discriminant Analysis based on DB-SMOTE Balanced Data

#Next, we will use the models we have trained using 
#DB-SMOTE Balanced Data to generate predictions on the test dataset.

##decision tree model - Trained on DB-SMOTE Balanced Data

dt_dbsmote_prediction <- predict(dt_dbsmote, test_data, type = "prob") #step 1: dt model prediction

dt_dbsmote_test <- factor(ifelse(dt_dbsmote_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_dtdbsmote <- posPredValue(dt_dbsmote_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_dtdbsmote <- sensitivity(dt_dbsmote_test, test_data$class, positive = 'yes')
F1_dtdbsmote <- (2*precision_dtdbsmote*recall_dtdbsmote)/(precision_dtdbsmote+recall_dtdbsmote)


##Naive Bayes Model - Trained on DB-SMOTE Balanced Data

nb_dbsmote_prediction <- predict(nb_dbsmote, test_data, type = "prob") #step 1: naive bayes model prediction

nb_dbsmote_test <- factor(ifelse(nb_dbsmote_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_nbdbsmote <- posPredValue(nb_dbsmote_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_nbdbsmote <- sensitivity(nb_dbsmote_test, test_data$class, positive = 'yes')
F1_nbdbsmote <- (2*precision_nbdbsmote*recall_nbdbsmote)/(precision_nbdbsmote+recall_nbdbsmote)

##LDA Model - Trained on DB-SMOTE Balanced Data

lda_dbsmote_prediction <- predict(lda_dbsmote, test_data, type = "prob") #step 1: naive bayes model prediction

lda_dbsmote_test <- factor(ifelse(lda_dbsmote_prediction$yes>0.50, 'yes', 'no')) #step 2: assign class to probabilities

precision_ldadbsmote <- posPredValue(lda_dbsmote_test, test_data$class, positive = 'yes') #step 3 (3 lines): measure performance with precision, recall, and F1
recall_ldadbsmote <- sensitivity(lda_dbsmote_test, test_data$class, positive = 'yes')
F1_ldadbsmote <- (2*precision_ldadbsmote*recall_ldadbsmote)/(precision_ldadbsmote+recall_ldadbsmote)

### Task 9(final task): Compare the model performance 

#We will compare the recall, precision, and F1 performance measures for 
#each of the three models we trained using the four training datasets: original imbalanced, SMOTE balanced, ADASYN balanced, and DB SMOTE balanced. 

#the most important performance measure for the fraudulent problem is the recall, which measures how complete our results are indicating the model captures more of the fraudulent transactions.


#Compare the Recall of the models: TP / TP + FN. To do that, we'll need to combine our results into a dataframe
model_compare_recall <- data.frame(Model = c('dt_orig',
                                             'nb_orig',
                                             'lda_orig',
                                             'dt_smote',
                                             'nb_smote',
                                             'lda_smote',
                                             'dt_adas',
                                             'nb_adas',
                                             'lda_adas',
                                             'dt_dbsmote',
                                             'nb_dbsmote',
                                             'lda_dbsmote' ),
                                   Recall = c(recall_dtorig,
                                              recall_nborig,
                                              recall_ldaorig,
                                              recall_dtsmote,
                                              recall_nbsmote,
                                              recall_ldasmote,
                                              recall_dtadas,
                                              recall_nbadas,
                                              recall_ldaadas,
                                              recall_dtdbsmote,
                                              recall_nbdbsmote,
                                              recall_ldadbsmote))

ggplot(aes(x=reorder(Model,-Recall) , y=Recall), data=model_compare_recall) +
  geom_bar(stat='identity', fill = 'light blue') +
  ggtitle('Comparative Recall of Models on Test Data') +
  xlab('Models')  +
  ylab('Recall Measure')+
  geom_text(aes(label=round(Recall,2)))+
  theme(axis.text.x = element_text(angle = 40))

#Compare the Precision of the models: TP/TP+FP [note update the names of the precision object if you used different names]
model_compare_precision <- data.frame(Model = c('dt_orig',
                                                'nb_orig',
                                                'lda_orig',
                                                'dt_smote',
                                                'nb_smote',
                                                'lda_smote',
                                                'dt_adas',
                                                'nb_adas',
                                                'lda_adas',
                                                'dt_dbsmote',
                                                'nb_dbsmote',
                                                'lda_dbsmote'),
                                      Precision = c(precision_dtorig,
                                                    precision_nborig,
                                                    precision_ldaorig,
                                                    precision_dtsmote,
                                                    precision_nbsmote,
                                                    precision_ldasmote,
                                                    precision_dtadas,
                                                    precision_nbadas,
                                                    precision_ldaadas,
                                                    precision_dtdbsmote,
                                                    precision_nbdbsmote,
                                                    precision_ldadbsmote))

ggplot(aes(x=reorder(Model,-Precision) , y=Precision), data=model_compare_precision) +
  geom_bar(stat='identity', fill = 'light green') +
  ggtitle('Comparative Precision of Models on Test Data') +
  xlab('Models')  +
  ylab('Precision Measure')+
  geom_text(aes(label=round(Precision,2)))+
  theme(axis.text.x = element_text(angle = 40))


#Compare the F1 of the models: 2*((Precision*Recall) / (Precision + Recall)) [note update the names of the F1 object if you used different names]
model_compare_f1 <- data.frame(Model = c('dt_orig',
                                         'nb_orig',
                                         'lda_orig',
                                         'dt_smote',
                                         'nb_smote',
                                         'lda_smote',
                                         'dt_adas',
                                         'nb_adas',
                                         'lda_adas',
                                         'dt_dbsmote',
                                         'nb_dbsmote',
                                         'lda_dbsmote'),
                               F1 = c(F1_dtorig,
                                      F1_nborig,
                                      F1_ldaorig,
                                      F1_dtsmote,
                                      F1_nbsmote,
                                      F1_ldasmote,
                                      F1_dtadas,
                                      F1_nbadas,
                                      F1_ldaadas,
                                      F1_dtdbsmote,
                                      F1_nbdbsmote,
                                      F1_ldadbsmote))

ggplot(aes(x=reorder(Model,-F1) , y=F1), data=model_compare_f1) +
  geom_bar(stat='identity', fill = 'red') +
  ggtitle('Comparative F1 of Models on Test Data') +
  xlab('Models')  +
  ylab('F1 Measure')+
  geom_text(aes(label=round(F1,2)))+
  theme(axis.text.x = element_text(angle = 40))



#The END






