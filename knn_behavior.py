#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 21:49:08 2025

@author: amber
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

behavior=pd.read_csv('/Users/amber/Desktop/data mining mism3515/project proposal/consumer behavior and shopping habits/shopping_behavior_updated.csv')

'''BEHAVIOR'''
frequency_map = {
    'Daily': 30,
    'Fortnightly': 2,     # every 2 weeks
    'Weekly': 4,        # about 7 times per month
    'Bi-Weekly': 2,
    'Monthly': 1,
    'Quarterly': 0.25,
    'Every 3 Months': 0.33,
    'Annually': 0.08,       # about once per 12 months
}


behavior['Frequency_Of_Purchases'] = behavior['Frequency of Purchases'].map(frequency_map)



behavior['Impulsive_Purchase'] = (
    ((behavior['Frequency_Of_Purchases'] >= 1) |  # monthly or more
     (behavior['Previous Purchases'] > behavior['Previous Purchases'].mean())) &
    ((behavior['Discount Applied'] == 'Yes') | (behavior['Promo Code Used'] == 'Yes'))
).astype(int)

behavior['Impulsive_Purchase'].value_counts(normalize=True)

behavior = behavior.drop(columns=['Customer ID', 'Frequency of Purchases','Color', 'Size', 
                                  'Item Purchased', 'Discount Applied', 'Promo Code Used'])


behavior = pd.get_dummies(behavior, drop_first=True).astype(int)

y_behavior = behavior['Impulsive_Purchase']


x_behavior = behavior.drop(columns='Impulsive_Purchase')

##********************************************************************

X = x_behavior
y = y_behavior

# ----------------------------------------

'''train splitting data'''

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3,random_state=1, stratify=y)


'''scaling the data'''
scaler = MinMaxScaler()        
scaler.fit(X_train)            

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''fitting the data'''
knn = KNeighborsClassifier()   
knn.fit(X_train_scaled, y_train)   

'''predicting'''

# Evaluate with F1
y_pred_test = knn.predict(X_train_scaled)
f1_basic_training_before = f1_score(y_train, y_pred_test)
# ==> 0.8048780487804879


y_pred = knn.predict(X_test_scaled)
f1_basic_testing_before = f1_score(y_test, y_pred)
print("F1 score (behavior):", f1_basic_testing_before)
### 0.6441631504922645


print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification report (baseline):")
print(classification_report(y_test, y_pred))

## [688  93]
## [160 229]]

#Precision ≈ 0.71

#Recall ≈ 0.59

# ----------------------------------------
# 5. Pipeline + GridSearchCV for KNN
# ----------------------------------------

# Pipeline: MinMaxScaler (step 'a') → KNN (step 'b')
pipe = Pipeline([
    ('a', MinMaxScaler()),
    ('b', KNeighborsClassifier())
])

# Fit the pipeline once with default KNN (optional baseline)
pipe.fit(X_train, y_train)

y_pred_pipe = pipe.predict(X_test)
f1_pipe = f1_score(y_test, y_pred_pipe)
print("Pipeline default KNN F1 score (behavior):", f1_pipe)
## 0.6441631504922645


# Hyperparameter grid: K and distance metric p
param_grid = {
    'b__n_neighbors': range(1, 20),   # K from 1 to 19
    'b__p': [1, 2]                    # p=1 (Manhattan), p=2 (Euclidean)
}

# Grid search with F1 scoring
grid = GridSearchCV(
    pipe,
    param_grid,
    verbose=3,
    scoring="f1",
    cv=5
)

# This may take a while
grid.fit(X_train, y_train)

###param_grid={'b__n_neighbors': range(1, 20), 'b__p': [1, 2]},
### scoring='f1', verbose=3)

print("Best parameters (behavior):", grid.best_params_)
## 'b__n_neighbors': 19, 'b__p': 1
print("Best cross-validated F1 (behavior):", grid.best_score_)

# ----------------------------------------
# 6. Final KNN model using best hyperparameters
# ----------------------------------------
best_k = grid.best_params_['b__n_neighbors']
best_p = grid.best_params_['b__p']
print("Best K:", best_k)
print("Best p:", best_p)

best_pipe = Pipeline([
    ('a', MinMaxScaler()),
    ('b', KNeighborsClassifier(n_neighbors=best_k, p=best_p))
])


best_pipe.fit(X_train, y_train)


y_pred_test = knn.predict(X_train_scaled)
f1_score(y_train, y_pred_test)


y_pred_best = best_pipe.predict(X_test)
f1_best = f1_score(y_test, y_pred_best)

print("Final tuned KNN F1 score on test set (behavior):", f1_best)
##0.6875934230194319


print("Confusion matrix (tuned KNN, behavior):")
print(confusion_matrix(y_test, y_pred_best))

print("Classification report (baseline):")
print(classification_report(y_test, y_pred_best))
##[[731  50]
## [159 230]]

#Precision ≈ 0.82

#Recall ≈ 0.59

#F1 ≈ 0.69

###short analysis:

##F1 improved from ~0.64 → ~0.69
#Precision for impulsive buyers jumped a lot (~0.71 → ~0.82)
#Recall for impulsive buyers stayed almost the same (~0.59 → ~0.59)

## --> After tuning K and the distance metric, the KNN model became more precise at 
#identifying impulsive buyers without sacrificing recall, which increased the 
#overall F1 score from 0.64 to 0.69.



import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate confusion matrix using your tuned predictions
cm = confusion_matrix(y_test, y_pred_best)

# Plot it
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["Not Impulsive", "Impulsive"],
    yticklabels=["Not Impulsive", "Impulsive"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Tuned KNN (Behavior Dataset)")
plt.tight_layout()
plt.show()