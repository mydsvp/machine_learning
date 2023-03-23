#!/usr/bin/env python
# coding: utf-8

# Matthew Yeseta
# * Email: aidatasmart@gmail.com
# * Master of Science, Data Science, Indiana University
# 
# **Binary Classification** using Multi Class Classification Developemtn and Performance Compaison using digits   
# MNIST Digit Recognizer public data. Tunee each Binary Classification using RandomizedSearchCV for best fitting hyperparameters with automated parmater assignment in classifer model. Developed the following customized Python Multi-Class classification strategies to research and observe how each binary classifiers (Binary Logistic, Binary Logistic Softmax Regression, Binary Support Vector with RBF Kernal, KNN KNeighbors Classifier for Binary analysis. 
#   
# **OneVsOne Classifiers**  
# 
# * Python LogisticRegression
# 
# * Python SVC with kernel RBF 
# 
# * Python LogisticRegression Softmax Regression, ie multi_class Mmultinomial
# 
# * Python KNeighborsClassifier 
#  
# 
# **OneVsRest Classifiers**
# 
# * Python LogisticRegression
# 
# * Python SVC with kernel RBF
# 
# * Python LogisticRegression Softmax Regression, ie multi_class Mmultinomial
# 
# * Python KNeighborsClassifier

# In[338]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, cross_val_score
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.svm import SVC


# #############################################################################################
# ## Python Logistic Regression model 
# ## OneVsOne Classifier, OneVsRest Binary Classifier base models
# #############################################################################################

# In[339]:


def SetLogisticOVCluster(X_train_scaled, y_train):
   # Define Logistic Regression model using OneVsOne Classifier and OneVsRest Classifier base models
    ovr_model = OneVsRestClassifier(LogisticRegression())
    ovo_model = OneVsOneClassifier(LogisticRegression())

    # Define common hyperparameters to tune Logistic Regression estimator 
    # The pass-through to Logistic estimator is estimator__
    param_grid = {
        'estimator__max_iter' : [2500, 4500, 6500, 9500, 14000],
            'estimator__C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 2.5, 3.0, 5, 10, 20]
    }    

    # Tune hyperparameters using RandomizedSearchCV
    ovr_grid_param = RandomizedSearchCV(ovr_model, param_grid, cv=5, n_jobs=-1, error_score="raise")
    ovo_grid_param = RandomizedSearchCV(ovo_model, param_grid, cv=5, n_jobs=-1, error_score="raise")

    # Use the best models to make predictions and evaluate performance
    ovr_fit = ovr_grid_param.fit(X_train_scaled, y_train) 
    ovo_fit = ovo_grid_param.fit(X_train_scaled, y_train)

    print("\n [Estimator] OneVsRest best estimator via Randomized Search:\n", ovr_fit.best_estimator_)
    print("\n [Score] OneVsRest best score via Randomized Search:\n", ovr_fit.best_score_)
    print("\n [Tuned] OneVsRest best tuned parameters provided by Randomized Search:\n", ovr_fit.best_params_)

    print("\n [Estimator] OneVsOne best estimator via Randomized Search:\n", ovo_fit.best_estimator_)
    print("\n [Score] OneVsOne best score via Randomized Search:\n", ovo_fit.best_score_)
    print("\n [Tuned] OneVsOne best tuned parameters provided by Randomized Search:\n", ovo_fit.best_params_)

    
    return ovr_fit.best_params_, ovo_fit.best_params_

       


# In[340]:



def FitModelLogisticOVCluster(X, y_train, X_test, X_train_scaled, ovr_params_, ovo_params_):
    ## Automate best OneVsRest Classifier Randomized Search tuned hyperparameters

    ovr_model_best = OneVsRestClassifier(
        LogisticRegression(multi_class='ovr',
                           C=ovr_params_['estimator__C'],
                           max_iter=ovr_params_['estimator__max_iter'], 
                           solver='liblinear'
                          )
    )
    ovr_model_best_fit = ovr_model_best.fit(X, y_train) 
    ovr_model_best_pred = ovr_model_best_fit.predict(X_test.iloc[:, 2:786])

    ovr_score_lr = cross_val_score(ovr_model_best, X, y_train, cv=3, scoring="accuracy")

    ## Automate best OneVsOne Classifier Randomized Search tuned hyperparameters

    ovo_model_best = OneVsOneClassifier(
        LogisticRegression(multi_class='ovr',
                           C=ovo_params_['estimator__C'], 
                           max_iter=ovo_params_['estimator__max_iter'], 
                           solver='liblinear'
                          )
    )
    ovo_model_best_fit = ovo_model_best.fit(X_train_scaled, y_train)
    ovo_model_best_pred = ovo_model_best_fit.predict(X_test.iloc[:, 2:786]) 

    ovo_score_lr = cross_val_score(ovo_model_best, X, y_train, cv=3, scoring="accuracy")
    
    ## Score neVsOne Classifier and OneVsRest Classifiers accuracy
    print("OneVsRestClassifier accuracy: {:.2f}%", format(ovr_score*100))
    print("OneVsOneClassifier accuracy: {:.2f}%", format(ovo_score*100))
    
    return ovr_model_best_pred, ovo_model_best_pred, ovr_score_lr, ovo_score_lr
    


# #############################################################################################
# ## Python Support Vector (RBF kernal) Classifer
# ## OneVsOne Classifier, OneVsRest Binary Classifier base models
# #############################################################################################

# In[341]:


def SetRbfOVCluster(X_train_scaled, y_train):

    # Define SVC RBF kernal OneVsOne Classifier and OneVsRest Classifier base models

    # Define SVC RBF using OneVsOne Classifier and OneVsRest Classifier
    svc_RBF_OVR_model = OneVsRestClassifier(SVC(kernel='rbf'))
    svc_RBF_OVO_model = OneVsOneClassifier(SVC(kernel='rbf'))

    # Define common hyperparameters to tune for SVC RBF Kernal estimator
    # The pass-through to SVC RBF Kernal estimator is estimator__
    param_grid = {
#        "estimator__gamma": ['scale', 'auto'],
        "estimator__C": [1,2,4,8],
        "estimator__kernel": ['rbf'],
        "estimator__degree":[1, 2, 3, 4]
    }
    # Tune the hyperparameters using RandomizedSearchCV
    svc_RBF_OVR_grid_param = RandomizedSearchCV(svc_RBF_OVR_model, param_grid, cv=5, n_jobs=3, error_score="raise")
    svc_RBF_OVO_grid_param = RandomizedSearchCV(svc_RBF_OVO_model, param_grid, cv=5, n_jobs=3, error_score="raise")

   
    svc_RBF_OVR_model_fit = svc_RBF_OVR_grid_param.fit(X_train_scaled, y_train) 
    svc_RBF_OVO_model_fit = svc_RBF_OVO_grid_param.fit(X_train_scaled, y_train) 

    # Best Tuned Hyperparaters 
    print("\n [Estimator] OneVsRest RBF SVC best estimator via Randomized Search:\n", svc_RBF_OVR_model_fit.best_estimator_)
    print("\n [Score] OneVsRest RBF SVC best score via Randomized Search:\n", svc_RBF_OVR_model_fit.best_score_)
    print("\n [Tuned] OneVsRest RBF SVC best parameters provided by Randomized Search:\n", svc_RBF_OVR_model_fit.best_params_)

    print("\n [Estimator] OneVsOne RBF SVC best estimator via Randomized Search:\n", svc_RBF_OVO_model_fit.best_estimator_)
    print("\n [Score] OneVsOne RBF SVC best score via Randomized Search:\n", svc_RBF_OVO_model_fit.best_score_)
    print("\n [Tuned] OneVsOne RBF SVC best parameters provided by Randomized Search:\n", svc_RBF_OVO_model_fit.best_params_)
    
    return svc_RBF_OVR_model_fit.best_params_, svc_RBF_OVO_model_fit.best_params_


# In[342]:


def FitModelRbfOVCluster(X, y_train, X_test, X_train_scaled, ovr_params_, ovo_params_):
    
    ## Automate best OneVsRest SVC RBF Classifier Randomized Search tuned hyperparameters
    svc_RBF_OVR_model_best = OneVsRestClassifier(
        SVC(gamma='auto',
            kernel='rbf', 
            C=ovr_params_['estimator__C']
           )
    )
    svc_RBF_OVR_model_best_fit = svc_RBF_OVR_model_best.fit(X, y_train)
    svc_RBF_OVR_model_best_pred = svc_RBF_OVR_model_best_fit.predict(X_test.iloc[:, 2:786])

    svc_RBF_OVR_score = cross_val_score(svc_RBF_OVR_model_best, X, y_train, cv=3, scoring="accuracy")

    ## Automate best OneVsOne SVC RBF Classifier Randomized Search tuned hyperparameters
    svc_RBF_OVO_model_best = OneVsOneClassifier(
        SVC(gamma='auto',
            kernel='rbf', 
            C=ovo_params_['estimator__C']
           )
    )
    svc_RBF_OVO_model_best_fit = svc_RBF_OVO_model_best.fit(X, y_train)
    svc_RBF_OVO_model_best_pred = svc_RBF_OVO_model_best_fit.predict(X_test.iloc[:, 2:786])

    svc_RBF_OVO_score = cross_val_score(svc_RBF_OVO_model_best, X, y_train, cv=3, scoring="accuracy")

    ## Score OneVsOne Classifier and OneVsRest Classifiers accuracy
    print("OneVsRestClassifier accuracy: {:.2f}%", format(svc_RBF_OVR_score*100))
    print("OneVsOneClassifier accuracy: {:.2f}%", format(svc_RBF_OVO_score*100))
    
    return svc_RBF_OVR_model_best_pred, svc_RBF_OVO_model_best_pred, svc_RBF_OVR_score, svc_RBF_OVO_score


# #############################################################################################
# ## Python Softmax Regression MultiNomial Model
# ## OneVsOne Classifier, OneVsRest Binary Classifier base models
# #############################################################################################

# In[343]:


def SetSoftmaxOVCluster(X_train_scaled, y_train):

    # Define Softmax Regression MultiNomial OneVsOne Classifier and OneVsRest Classifier base models

    # Define Softmax Regression MultiNomial using OneVsOne Classifier and OneVsRest Classifier
    ovr_softmax_model = OneVsRestClassifier(LogisticRegression(multi_class = 'multinomial'))
    ovo_softmax_model = OneVsOneClassifier(LogisticRegression(multi_class = 'multinomial'))

    # Define common hyperparameters to tune for Softmax Regression MultiNomial estimator
    # The pass-through to SVC RBF Kernal estimator is estimator__
    param_grid = {
        'estimator__max_iter' : [2000, 2500, 3000, 4000, 6500, 9500, 14000, 20000, 30000],
        'estimator__C': [0.05, 0.1, 0.5, 1.0, 1.5, 2, 3, 5]
    }
    
    # Tune the hyperparameters using GridSearchCV
    ovr_softmax_grid_param = RandomizedSearchCV(ovr_softmax_model, param_grid, cv=5, n_jobs=3, error_score="raise")
    ovo_softmax_grid_param = RandomizedSearchCV(ovo_softmax_model, param_grid, cv=5, n_jobs=3, error_score="raise") 
    
    ## score tuning model
    ovr_softmax_model_model_fit = ovr_softmax_grid_param.fit(X_train_scaled, y_train) 
    ovo_softmax_model_model_fit = ovo_softmax_grid_param.fit(X_train_scaled, y_train) 

    # Best Tuned Hyperparamters

    print("\n [Estimator] OneVsRest Softmax best estimator via Randomized Search:\n", ovr_softmax_model_model_fit.best_estimator_)
    print("\n [Score] OneVsRest Softmax best score via Randomized Search:\n", ovr_softmax_model_model_fit.best_score_)
    print("\n [Tuned] OneVsRest Softmax best parameters provided by Randomized Search:\n", ovr_softmax_model_model_fit.best_params_)

    print("\n [Estimator] OneVsOne Softmax best estimator across Randomized Search:\n", ovo_softmax_grid_param.best_estimator_)
    print("\n [Score] OneVsOne Softmax best score across Randomized Search:\n", ovo_softmax_grid_param.best_score_)
    print("\n [Tuned] OneVsOne Softmax best parameters across Randomized Search:\n", ovo_softmax_grid_param.best_params_)

    return ovr_softmax_model_model_fit.best_params_, ovo_softmax_grid_param.best_params_


# In[344]:


def FitModelSoftmaxOVCluster(X, y_train, X_test, X_train_scaled, ovr_params_, ovo_params_):

    ## Automate best OneVsRest LogisticRegression Multinomial softmax using Randomized Search tuned hyperparameters

    ovr_softmax_model_best = OneVsRestClassifier(
        LogisticRegression(multi_class = 'multinomial', 
                           max_iter=ovr_params_['estimator__max_iter'],                                          
                           C=ovr_params_['estimator__C']
                          )
    )
    ovr_softmax_model_best_fit = ovr_softmax_model_best.fit(X, y_train)
    ovr_softmax_model_best_pred = ovr_softmax_model_best_fit.predict(X_test.iloc[:, 2:786])

    ovr_softmax_score = cross_val_score(ovr_softmax_model_best, X, y_train, cv=3, scoring="accuracy")

    ## Automate best OneVsOne LogisticRegression Multinomial softmax using Randomized Search tuned hyperparameters
    ovo_softmax_model_best = OneVsOneClassifier(
        LogisticRegression(multi_class = 'multinomial',
                           max_iter=ovo_params_['estimator__max_iter'], 
                           C=ovo_params_['estimator__C']
                          )
    )
    ovo_softmax_model_best_fit = ovo_softmax_model_best.fit(X, y_train)
    ovo_softmax_model_best_pred = ovo_softmax_model_best_fit.predict(X_test.iloc[:, 2:786])

    ovo_softmax_score = cross_val_score(ovo_softmax_model_best, X, y_train, cv=3, scoring="accuracy")

    ## Score OneVsOne Classifier and OneVsRest Classifiers accuracy
    print("OneVsRestClassifier accuracy: {:.2f}%", format(ovr_softmax_score*100))
    print("OneVsOneClassifier accuracy: {:.2f}%", format(ovo_softmax_score*100))
    
    return ovr_softmax_model_best_pred, ovo_softmax_model_best_pred, ovr_softmax_score, ovo_softmax_score 
    


# #############################################################################################
# ## Python Neighbors KNN Regression Model
# ## OneVsOne Classifier, OneVsRest Binary Classifier base models
# #############################################################################################

# In[345]:


def SetKnnOVClassifier(X_train_scaled, y_train):
    # Define KNeighbors KNN Regression OneVsOne Classifier and OneVsRest Classifier base models

    # Define KNN using OneVsOne Classifier and OneVsRest Classifier
    ovr_KNN_model = OneVsRestClassifier(KNeighborsClassifier())
    ovo_KNN_model = OneVsOneClassifier(KNeighborsClassifier())

    # Define common hyperparameters to tune for KNN estimator
    # The pass-through to KNN estimator is estimator__
    param_grid = {
        'estimator__n_neighbors': [1,2,3,4,5,6,8,10,11,12,14,20],
        'estimator__weights': ('uniform', 'distance'),
        'estimator__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute', 'auto'),
        'estimator__leaf_size': [10,20,30,40,50,60,80,120]
    }

    # Tune the hyperparameters using RandomizedSearchCV
    ovr_KNN_grid_param = RandomizedSearchCV(ovr_KNN_model, param_grid, cv=5, n_jobs=3, error_score="raise")
    ovo_KNN_grid_param = RandomizedSearchCV(ovo_KNN_model, param_grid, cv=5, n_jobs=3, error_score="raise")
    
    ## score tuning
    ovr_KNN_model_model_fit = ovr_KNN_grid_param.fit(X_train_scaled, y_train) 
    ovo_KNN_model_model_fit = ovo_KNN_grid_param.fit(X_train_scaled, y_train) 

    # Best Tuned Hyperparamters

    print("\n OneVsRest KNN best estimator via Randomized Search:\n", ovr_KNN_model_model_fit.best_estimator_)
    print("\n OneVsRest KNN best score via Randomized Search:\n", ovr_KNN_model_model_fit.best_score_)
    print("\n OneVsRest KNN best parameters provided by Randomized Search:\n", ovr_KNN_model_model_fit.best_params_)

    print("\n OneVsOne KNN best estimator across Randomized Search:\n", ovo_KNN_grid_param.best_estimator_)
    print("\n OneVsOne KNN best score across Randomized Search:\n", ovo_KNN_grid_param.best_score_)
    print("\n OneVsOne KNN best parameters provided by Randomized Search:\n", ovo_KNN_grid_param.best_params_)
    
    return ovr_KNN_model_model_fit.best_params_, ovo_KNN_grid_param.best_params_


# In[346]:


def FitModelKnnOVClassifier(X, y_train, X_test, X_train_scaled, ovr_params_, ovo_params_):

    ## OneVsRest Classifier KNN KNeighborsClassifier model using best parameters
    nn = ovr_params_['estimator__n_neighbors']
    ovr_KNN_model_best = OneVsRestClassifier(
        KNeighborsClassifier(n_neighbors=nn, #ovr_params_['estimator__n_neighbors'], 
                                weights=ovr_params_['estimator__weights'], 
                                algorithm=ovr_params_['estimator__algorithm'], 
                                leaf_size=ovr_params_['estimator__leaf_size'])
    )
    ovr_KNN_model_best_fit = ovr_KNN_model_best.fit(X, y_train)
    ovr_KNN_model_best_pred = ovr_KNN_model_best_fit.predict(X_test.iloc[:, 2:786])

    ovr_KNN_score = cross_val_score(ovr_KNN_model_best, X, y_train, cv=3, scoring="accuracy")

    ## OneVsRest Classifier KNN KNeighborsClassifier model using best parameters
    nn = ovo_params_['estimator__n_neighbors']
    ovo_KNN_model_best = OneVsOneClassifier(
        KNeighborsClassifier(n_neighbors=nn, #ovo_params_['estimator__n_neighbors'], 
                                weights=ovo_params_['estimator__weights'], 
                                algorithm=ovo_params_['estimator__algorithm'], 
                                leaf_size=ovo_params_['estimator__leaf_size']))
    ovo_KNN_model_best_fit = ovo_KNN_model_best.fit(X, y_train)
    ovo_KNN_model_best_pred = ovo_KNN_model_best_fit.predict(X_test.iloc[:, 2:786])

    ovo_KNN_score = cross_val_score(ovo_KNN_model_best, X, y_train, cv=3, scoring="accuracy")
    
    ## Score OneVs Classifier and OneVsRest Classifiers accuracy
    print("OneVsRestClassifier accuracy: {:.2f}%", format(ovr_KNN_score*100))
    print("OneVsOneClassifier accuracy: {:.2f}%", format(ovo_KNN_score*100))
    
    return ovr_KNN_model_best_pred, ovo_KNN_model_best_pred, ovr_KNN_score, ovo_KNN_score


# #############################################################################################
# ## Python Function for all OneVsOne, OneVsRest Binary Classifier models
# #############################################################################################

# In[347]:


def writeModelPrediction(prediction_model, filename):

    ### save for Kaggle submission
    submission = pd.concat([pd.Series(X_train['ImageId']),pd.Series(prediction_model)],axis=1, ignore_index=True)

    submission.columns = ['ImageId', 'Label']

    submission = submission[(submission.iloc[:,1].values > 0)] 
    submission = submission[(submission.iloc[:,0].values > 0)]

    submission.to_csv(filename,index=False, header=1)  
    


# In[348]:


def SetTrainTest(train):
    ## split train / test    
    train = train.dropna()

    y = train['label']
    X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.60, random_state=42)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train.iloc[:, 2:786])
    X = X_train.iloc[:, 2:786]
    y = X_train['label']

    #X_train = X_train.drop(['label'], axis=1)
    
    return X_train_scaled, X_test, y_train, y_test, X, y
  


# In[349]:


def SetImageId(train):
    image_Id = np.arange(1, len(train))
    image_Id = image_Id.reshape(-1, 1)
    dfImageIds = pd.DataFrame(image_Id.astype('int64'), columns=['ImageId'])
    train = dfImageIds.join(train)  

    return train


# #############################################################################################
# ## Python Drive Function for OneVsOne, OneVsRest Prediction Accuracy
# #############################################################################################

# In[350]:


def PredictModel(X_train_scaled, X_test, y_train, y_test, X, y):
    ovr, ovo = SetLogisticOVCluster(X_train_scaled, y_train)
    ovr_soft_pred, ovo_soft_pred, ovr_score_lr, ovo_score_lr = FitModelSoftmaxOVCluster(X, y_train, X_test, X_train_scaled, ovr, ovo)

    SetRbfOVCluster(X_train_scaled, y_train)
    ovr_rbf_pred, ovo_rbf_pred, ovr_RBF_score, ovo_RBF_score = FitModelRbfOVCluster(X, y_train, X_test, X_train_scaled, ovr, ovo)
    
    SetSoftmaxOVCluster(X_train_scaled, y_train)
    ovr_sm_pred, ovo_sm_pred, ovr_sm_score, ovo_sm_score = FitModelSoftmaxOVCluster(X, y_train, X_test, X_train_scaled, ovr, ovo)
    
    SetKnnOVClassifier(X_train_scaled, y_train)
    ovr_knn_pred, ovr_knn_pred, ovr_KNN_score, ovo_KNN_score = FitModelKnnOVClassifier(X, y_train, X_test, X_train_scaled, ovr, ovo)
           
    ## Score OneVsOne Classifier and OneVsRest Classifiers accuracy
    print("Accuracy of Linear OneVsRestClassifier: {:.2f}%", format(ovr_score_lr*100))
    print("Accuracy of Linear OneVsOneClassifier: {:.2f}%", format(ovo_score_lr*100))

    ## Score OneVsOne Classifier and OneVsRest Classifiers accuracy
    print("Accuracy of RBF OneVsRestClassifier: {:.2f}%", format(ovr_RBF_score*100))
    print("Accuracy of RBF OneVsOneClassifier: {:.2f}%", format(ovo_RBF_score*100))

    ## Score OneVsOne Classifier and OneVsRest Classifiers accuracy
    print("Accuracy of Softmax OneVsRestClassifier: {:.2f}%", format(ovr_sm_score*100))
    print("Accuracy of Softmax OneVsOneClassifier: {:.2f}%", format(ovo_sm_score*100))

    ## Score OneVsOne Classifier and OneVsRest Classifiers accuracy
    print("Accuracy of KNN OneVsRestClassifier:  {:.2f}%", format(ovr_KNN_score*100))
    print("Accuracy of KNN OneVsOneClassifier: {:.2f}%", format(ovo_KNN_score*100))

    ovr_predict = list(ovr_score_lr, ovr_RBF_score, ovr_sm_score, ovr_KNN_score)
    ovo_predict = list(ovo_score_lr, ovo_RBF_score, ovo_sm_score, ovo_KNN_score)
    
    return ovr_predict, ovo_predict


# #############################################################################################
# ## Python Program Driver Function 
# #############################################################################################

# In[351]:


def main_driver():
    train = pd.read_csv('../data/DigitRecognizer/train.csv')
    baseline = train
    
    train = train.loc[(train['label'] < 2)]
    train = SetImageId(train)
    
    X_train_scaled, X_test, y_train, y_test, X, y = SetTrainTest(train)
    ovr_predict, ovo_predict = PredictModel(X_train_scaled, X_test, y_train, y_test, X, y)
    filename = '../data/DigitRecognizer/digits_submission_function_01.csv'
    writeModelPrediction(prediction_model, filename)

    train = baseline
    train = train.loc[(train['label'] > 1) & (train['label'] < 4)]
    train = SetImageId(train)
    X_train_scaled, X_test, y_train, y_test, X, y = SetTrainTest(train)
    ovr_predict, ovo_predict = PredictModel(X_train_scaled, X_test, y_train, y_test, X, y)
    filename = '../data/DigitRecognizer/digits_submission_function_23.csv'
    writeModelPrediction(prediction_model, filename)

    train = baseline
    train = train.loc[(train['label'] > 3) & (train['label'] < 6)]
    train = SetImageId(train)
    X_train_scaled, X_test, y_train, y_test, X, y = SetTrainTest(train)
    ovr_predict, ovo_predict = PredictModel(X_train_scaled, X_test, y_train, y_test, X, y)
    filename = '../data/DigitRecognizer/digits_submission_function_45.csv'
    writeModelPrediction(prediction_model, filename)

    train = baseline
    train = train.loc[(train['label'] > 5) & (train['label'] < 8)]
    train = SetImageId(train)
    X_train_scaled, X_test, y_train, y_test, X, y = SetTrainTest(train)
    ovr_predict, ovo_predict = PredictModel(X_train_scaled, X_test, y_train, y_test, X, y)
    filename = '../data/DigitRecognizer/digits_submission_function_67.csv'
    writeModelPrediction(prediction_model, filename)

    train = baseline
    train = train.loc[(train['label'] > 7) & (train['label'] < 10)]
    train = SetImageId(train)
    X_train_scaled, X_test, y_train, y_test, X, y = SetTrainTest(train)
    ovr_predict, ovo_predict = PredictModel(X_train_scaled, X_test, y_train, y_test, X, y)
    filename = '../data/DigitRecognizer/digits_submission_function_89.csv'
    writeModelPrediction(prediction_model, filename)


main_driver()




# In[ ]:





# In[ ]:




