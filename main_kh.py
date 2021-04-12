# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:33:28 2021

@author: Kevin Hennessey
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

## Read data from public github repository
features = pd.read_csv("C:/Users/khenn/OneDrive/Documents/SP2021/CSE5243/Data Challenge/train_features.csv")
demos = pd.read_csv("C:/Users/khenn/OneDrive/Documents/SP2021/CSE5243/Data Challenge/train_demos.csv")
labels = pd.read_csv("C:/Users/khenn/OneDrive/Documents/SP2021/CSE5243/Data Challenge/train_labels.csv")

#%%
## Pre-processing
## Vital measurement extraction: heart rate, blood pressure, respiratory rate, oxygen saturation
labels['mortality'].value_counts()

demos['age'].value_counts()
demos['ethnicity'].value_counts()
demos['admission_type'].value_counts()
demos['marital_status'].value_counts()
demos['gender'].value_counts()
demos['insurance'].value_counts()

## Select age, admission_type, gender, insurance for demographic variables
demos.iloc[:,[0,1,2,3,5]]
demo_df = pd.DataFrame(data = demos.iloc[:,[0,1,2,3,5]]).set_index('adm_id')
demo_df = pd.get_dummies(demo_df, columns = demo_df.columns)

## Extract minimum, maximum and mean values for heartrate, mean arterial bp, respiratory rate, and oxygen saturation
features_mean = pd.DataFrame(features.groupby("adm_id")[['heartrate', 'meanbp', 'resprate', 'spo2']].mean())
features_mean.columns = ['heartrate_mean', 'meanbp_mean', 'resprate_mean', 'spo2_mean']
features_max = pd.DataFrame(features.groupby("adm_id")[['heartrate', 'meanbp', 'resprate', 'spo2']].max())
features_max.columns = ['heartrate_max', 'meanbp_max', 'resprate_max', 'spo2_max']
features_min = pd.DataFrame(features.groupby("adm_id")[['heartrate', 'meanbp', 'resprate', 'spo2']].min())
features_min.columns = ['heartrate_min', 'meanbp_min', 'resprate_min', 'spo2_min']


## Merge max, min, and mean dataframes
feat_summary = features_mean.merge(features_max, on = 'adm_id', how = 'left')
feat_summary = feat_summary.merge(features_min, on = 'adm_id', how = 'left')
## Merge features with demo info
feature_df = feat_summary.merge(demo_df, on = 'adm_id', how = 'left')
feature_df.dtypes

#%%
## Left join features and labels
full_df = feature_df.merge(labels, on = "adm_id", how = 'left')
full_df = full_df.set_index('adm_id')

## Drop NaN values
full_df = full_df.dropna()


## Create Training and validation set
X_train, X_valid, y_train, y_valid = train_test_split(full_df.iloc[:,:-1], full_df['mortality'], test_size=0.2, random_state=1)

#%% Results and Comparison 

## Function to print statistice relating to model efficieny and correctness
def results_function(test_target, pred_target, model_name, model):
    ## Calculate model evalutaion statistics
    ##print(model_name, "best parameters:", model.best_params_)
    cm = metrics.confusion_matrix(test_target, pred_target)
    TN = cm[0][0]
    FP = cm[0][1]
    print(model_name, "Accuracy:",metrics.accuracy_score(test_target, pred_target))
    print(model_name, "Precision:", metrics.precision_score(test_target, pred_target))
    print(model_name, "Specificity:", TN/(TN + FP))
    print(model_name, "Recall:", metrics.recall_score(test_target, pred_target))
    print(model_name, "F1 Score:", metrics.f1_score(test_target, pred_target))
    
## Function to plot ROC curve as well as calculate the AUC 
def roc(test_target, probs, model_name):
    ## Calculate the true positive rate and false positive rate
    fpr, tpr, thresholds = metrics.roc_curve(test_target, probs[:,1])
    ## Calcualte the ROC area under the curve
    roc_auc = metrics.auc(fpr, tpr)
    ## Plot the ROC curve
    plt.title('ROC '+ model_name)
    plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.draw()


#%%
## Build a decision Tree classifier
dtree = DecisionTreeClassifier()

### Feature Vetor 1
# Train Decision Tree Classifer
dtree.fit(X_train,y_train)
# Predict the response for test dataset
y_pred_dtree = dtree.predict(X_valid)
dtree_prob = dtree.predict_proba(X_valid)

## Feature Vector 1 Results
results_function(y_valid, y_pred_dtree, "Decision Tree", dtree)
roc(y_valid, dtree_prob, "Decision Tree")
disp_dtree = metrics.plot_confusion_matrix(dtree, X_valid, y_valid,cmap=plt.cm.Blues)
disp_dtree.ax_.set_title("Decision Tree")
plt.show()
#%%
## Build a Logisitc Regression Model
logRegr = LogisticRegression(max_iter = 1000)

logRegr.fit(X_train,y_train)
y_pred_log = logRegr.predict(X_valid)
log_prob = logRegr.predict_proba(X_valid)

## Feature Vector 1 Results
results_function(y_valid, y_pred_log, "Logisitc Regression", logRegr)
roc(y_valid, log_prob, "Logistic Regression")
disp_log = metrics.plot_confusion_matrix(logRegr, X_valid, y_valid,cmap=plt.cm.Blues)
disp_log.ax_.set_title("Logistic Regression")
plt.show()
#%% Classifier 2
## Naive Bayes Classifier
## 1st experimental dataset 
nb = MultinomialNB()

# No extra hyper parameters specified for Naive Bayes model
#params = {}

## Utilize GreadSearchCV to perform cross validation, CV = 5
#grid_nb = GridSearchCV(estimator = nb,
                       # param_grid = params, 
                       #cv = 5, 
                       # scoring = 'accuracy',
                       # verbose = 3,
                       # return_train_score = True,
                       # n_jobs = -1)


### Feature Vector 1
# Train Naive Bayes Model
nb.fit(X_train, y_train)
# predict the response and Probability resoponse
y_pred_nb = nb.predict(X_valid)
nb_prob1 = nb.predict_proba(X_valid)

## Feature Vector 1 Results
results_function(y_valid, y_pred_nb, "Naive Bayes Model", nb)
roc(y_valid, nb_prob1, "Naive Bayes Model")
## Plot Confusion Matrix
disp_nb = metrics.plot_confusion_matrix(nb, X_valid, y_valid,cmap=plt.cm.Blues)
disp_nb.ax_.set_title("Naive Bayes Model")
plt.show()

#%% Classifier 3
## KNN
## Build KNN classifier
knn = KNeighborsClassifier()

## Specify parameters to perform model selection
# N_neightbors is set from 5, 10, 25, 40
# weights is set to distance 
# p= 1 is equivalent to manhattan distance, and p= 2 is euclidean distance
# Algorithm: auto will choose algorithm which best suits our data.
params = {
    'n_neighbors' : [3, 5, 10, 25],
    'weights': ['distance'],
    'p':[1, 2],
    'algorithm': ['auto']
}

## Utilize gridsearch to perform 5-fold cross validation on all
## available KNN models given our parameter set
grid_knn = GridSearchCV(estimator = knn,
                        param_grid = params,
                        scoring ='accuracy', 
                        cv = 5, 
                        return_train_score = True,
                        verbose = 3,
                        n_jobs = -1)
### Feature Vector 1
## Fit the training data
knn = grid_knn.fit(X_train, y_train)
## Predict on the test set
y_pred_knn = knn.predict(X_valid)
knn_prob1 = knn.predict_proba(X_valid)

### Feature Vector 1 Results
results_function(y_valid, y_pred_knn, "KNN Model", knn)
roc(y_valid, knn_prob1, "KNN Model")
## plot Confusion matrix
disp_knn = metrics.plot_confusion_matrix(knn, X_valid, y_valid,cmap=plt.cm.Blues)
disp_knn.ax_.set_title("KNN Model")
plt.show()
