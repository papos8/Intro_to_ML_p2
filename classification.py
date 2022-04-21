import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
from project1 import data_import
from sklearn.dummy import DummyClassifier
from sklearn import model_selection
from sklearn.model_selection import KFold
from toolbox_02450 import mcnemar
from toolbox_02450 import feature_selector_lr
#Importing X matrix from project1
X_raw = data_import()
attributeNames = ["Age", "Systolic BP", "Diastolic BP", "Blood Glucose", 
                  "Body Temperature", "Heart Rate", "Risk Level"]
#Prepare the data - Input/Output
y = X_raw[:,6]
X = X_raw[:,0:6]# configure the cross-validation procedure
N, M = X.shape

CV = model_selection.KFold(10, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)

#Initializing error variable
errors = []

# Initializing Classifiers
clf1 = LogisticRegression(solver='newton-cg',
                          random_state=1)
clf2 = KNeighborsClassifier(algorithm='ball_tree',
                            leaf_size=50)
clf3 = DummyClassifier(strategy = "most_frequent")

# Building the pipelines
pipe1 = Pipeline([('std', StandardScaler()),
                  ('clf1', clf1)])

pipe2 = Pipeline([('std', StandardScaler()),
                  ('clf2', clf2)])
pipe3 = Pipeline([('clf3', clf3)])


# Setting up the parameter grids
param_grid1 = [{'clf1__penalty': ['l2'],
                'clf1__C': np.power(10., np.arange(-4, 4))}]

param_grid2 = [{'clf2__n_neighbors': list(range(1, 10)),
                'clf2__p': [1, 2]}]
param_grid3 = [{}]
Features = np.zeros((M,10))
# Setting up multiple GridSearchCV objects, 1 for each algorithm
gridcvs = {}
inner_cv = KFold(n_splits=10, shuffle=True, random_state=1)

for pgrid, est, name in zip((param_grid1, param_grid2, param_grid3),
                            (pipe1, pipe2, pipe3),
                            ('Logistic Regression', 'KNN', 'Baseline')):
    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring='accuracy',
                       n_jobs=-1,
                       cv=inner_cv,
                       verbose=0,
                       refit=True)
    gridcvs[name] = gcv
    
    
y_true = []
yhat = []
#Next, we define the outer loop
#The training folds from the outer loop will be used in the inner loop for model tuning
#The inner loop selects the best hyperparameter setting
#This best hyperparameter setting can be evaluated on both the avg. over the inner test folds and the 1 corresponding test fold of the outer loop
for name, gs_est in sorted(gridcvs.items()):
   
    print(50 * '-', '\n')
    print('Algorithm:', name)
    print('    Inner loop:')
    
    outer_scores = []
    K=10
    outer_cv = KFold(n_splits=K, shuffle=True, random_state=1)
    
    #Initialize error
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    
    k = 0
    for train_idx, valid_idx in outer_cv.split(X, y):
          
        # Compute squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

        
        dy = []
        m = gridcvs[name].fit(X[train_idx], y[train_idx]) # run inner loop hyperparam tuning
        print('\n        Best ACC (avg. of inner test folds) %.2f%%' % (gridcvs[name].best_score_ * 100))
        print('        Best parameters:', gridcvs[name].best_params_)
        if name == 'Logistic Regression':
            
            textout = ''
        
            selected_features, features_record, loss_record = feature_selector_lr(X[train_idx], y[train_idx], 10,display=textout)
        
            Features[selected_features,k] = 1
            print(selected_features)
        
        # Compute error rate
        
        errors.append(np.sum(m.predict(X[valid_idx])!=y[valid_idx])/len(y[valid_idx]))
        
        # perf on test fold (valid_idx)
        outer_scores.append(gridcvs[name].best_estimator_.score(X[valid_idx], y[valid_idx]))
        print('        ACC (on outer test fold) %.2f%%' % (outer_scores[-1]*100))
        
        yhat.append(m.predict(X[valid_idx]))
        y_true.append(y[valid_idx])
        k+=1
    
    print('\n    Outer Loop:')
    print('        ACC %.2f%% +/- %.2f' % 
              (np.mean(outer_scores) * 100, np.std(outer_scores) * 100))
    

yhat_bs = yhat[:10]
yhat_kn = yhat[10:20]
yhat_lr = yhat[20:30]
y_true = y_true[:10]

alpha = 0.05

print("KNN-Baseline")
for i in range(10):
    [thetahat, CI, p] = mcnemar(y_true[i], yhat_kn[i], yhat_bs[i], alpha=alpha)
    

print()
print("KNN-Logistic Regression")    
for i in range(10):
    [thetahat, CI, p] = mcnemar(y_true[i], yhat_kn[i], yhat_lr[i], alpha=alpha)
    

print()
print("Logistic Regression-Baseline")    
for i in range(10):
    [thetahat, CI, p] = mcnemar(y_true[i], yhat_lr[i], yhat_bs[i], alpha=alpha)
    
    
    
    
rLogisticRegression = LogisticRegression(C=1/100)
rLogisticRegression.fit(X_train,y_train)
p = [[45,110,80,10,100,90],[44,100,70,7,120,90],[50,160,100,10,120,80],[60,110,90,10,80,80]]
prediction = rLogisticRegression.predict(p)


#Select from the above analysis the best hyperparameters, define them and then run again
# param_grid2 = [{'n_neighbors': list(range(1, 10)),
#                 'p': [1, 2]}]



# gcv_model_select = GridSearchCV(estimator=clf3,
#                                 param_grid=param_grid3,
#                                 scoring='accuracy',
#                                 n_jobs=-1,
#                                 cv=inner_cv,
#                                 verbose=1,
#                                 refit=True)

# gcv_model_select.fit(X_train, y_train)
# print('Best CV accuracy: %.2f%%' % (gcv_model_select.best_score_*100))
# print('Best parameters:', gcv_model_select.best_params_)

# ## We can skip the next step because we set refit=True
# ## so scikit-learn has already fit the model to the
# ## whole training set

# # gcv_model_select.fit(X_train, y_train)

# train_acc = accuracy_score(y_true=y_train, y_pred=gcv_model_select.predict(X_train))
# test_acc = accuracy_score(y_true=y_test, y_pred=gcv_model_select.predict(X_test))

# print('Training Accuracy: %.2f%%' % (100 * train_acc))
# print('Test Accuracy: %.2f%%' % (100 * test_acc))


# #BASELINE MODEL

# #Calculate biggest class



