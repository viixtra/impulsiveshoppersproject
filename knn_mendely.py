#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 21:44:49 2025

@author: amber
"""

'''KNN MODELING FOR MENDELY'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report

##***********************************************************************************************************##
mendely=pd.read_csv('/Users/amber/Desktop/data mining mism3515/project proposal/mendely/Raw Data_Paylater and Non Paylater User.csv', sep=';')

mendely['SelfControl'] = mendely[['SC1', 'SC2','SC3 ', 'SC4 ', 'SC5']].mean(axis=1)
mendely['Happiness'] = mendely[['H1', 'H2', 'H3', 'H4']].mean(axis=1)
mendely['SocialInfluence'] = mendely[['SI1', 'SI2','SI3', 'SI4', 'SI5', 'SI6']].mean(axis=1)
mendely['Promotion'] = mendely[['P1', 'P2', 'P3', 'P4']].mean(axis=1)
mendely['NormalEval'] = mendely[['NE1', 'NE2', 'NE3', 'NE4', 'NE5']].mean(axis=1)
mendely['IBB_Score'] = mendely[['IBB1','IBB2','IBB3','IBB4']].mean(axis=1)


mendely['Impulsive_Buyer'] = (mendely['IBB_Score'] >= 3.5).astype(int)


mendely['Impulsive_Buyer'].value_counts(normalize=True)


likert_items_mendely = ['IBB1', 'IBB2', 'IBB3', 'IBB4', 'P1', 'P2', 'P3', 'P4', 'SI1', 'SI2',
'SI3', 'SI4', 'SI5', 'SI6', 'H1', 'H2', 'H3', 'H4', 'SC1', 'SC2',
'SC3 ', 'SC4 ', 'SC5', 'NE1', 'NE2', 'NE3', 'NE4', 'NE5']




mendely.drop(columns=likert_items_mendely)

mendely = mendely.drop(columns=['IBB1', 'IBB2', 'IBB3', 'IBB4', 'P1', 'P2', 'P3', 'P4', 'SI1', 'SI2',
                                'SI3', 'SI4', 'SI5', 'SI6', 'H1', 'H2', 'H3', 'H4', 'SC1', 'SC2',
                                'SC3 ', 'SC4 ', 'SC5', 'NE1', 'NE2', 'NE3', 'NE4', 'NE5'])

mendely = pd.get_dummies(mendely, drop_first=True)

x_mendely = mendely.drop(columns=['IBB_Score', 'Impulsive_Buyer'])

y_mendely=mendely['Impulsive_Buyer']

##***********************************************************************************************************##


# -------------------------
# 2. Define X and y
# -------------------------
X = x_mendely
y = y_mendely

# -------------------------
# 3. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1,
    stratify=y
)

# -------------------------
# 4. Manual MinMax scaling + baseline KNN
# -------------------------
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline KNN
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)


# Evaluate
y_pred_train = knn.predict(X_train_scaled)
f1_score(y_train, y_pred_train)
## 0.6691176470588235


y_pred = knn.predict(X_test_scaled)
f1_basic = f1_score(y_test, y_pred)
print("Baseline KNN F1 (Mendely):", f1_basic) ## 0.4628099173553719


print("Confusion matrix (baseline):")
print(confusion_matrix(y_test, y_pred))
#[[150  25]
# [ 40  28]]

print("Classification report (tuned):")
print(classification_report(y_test, y_pred))

# -------------------------
# 5. Pipeline + GridSearchCV
# -------------------------
pipe = Pipeline([
    ('a', MinMaxScaler()),
    ('b', KNeighborsClassifier())
])

pipe.fit(X_train, y_train)
y_pred_pipe = pipe.predict(X_test)
print("Pipeline baseline KNN F1 (Mendely):", f1_score(y_test, y_pred_pipe)) ## 0.4628099173553719

# Hyperparameter grid
param_grid = {
    'b__n_neighbors': range(1, 20),
    'b__p': [1, 2]     # Manhattan (1) vs Euclidean (2)
}

grid = GridSearchCV(
    pipe,
    param_grid,
    verbose=3,
    scoring="f1",
    cv=5
)

grid.fit(X_train, y_train)

print("Best parameters (Mendely):", grid.best_params_)
print("Best cross-validated F1:", grid.best_score_)
### 'b__n_neighbors': 5, 'b__p': 1

# -------------------------
# 6. Final tuned model
# -------------------------
best_k = grid.best_params_['b__n_neighbors']
best_p = grid.best_params_['b__p']
print("Best K:", best_k)
print("Best p:", best_p)

best_pipe = Pipeline([
    ('a', MinMaxScaler()),
    ('b', KNeighborsClassifier(n_neighbors=best_k, p=best_p))
])

best_pipe.fit(X_train, y_train)

# Predict on training
y_train_pred_best = best_pipe.predict(X_train)
f1_train_best = f1_score(y_train, y_train_pred_best)
print("Training F1 (Mendely, tuned):", f1_train_best)

# Predict on test
y_test_pred_best = best_pipe.predict(X_test)
f1_best = f1_score(y_test, y_test_pred_best)
print("Final tuned KNN F1 (Mendely, test):", f1_best)

print("Confusion matrix (tuned):")
print(confusion_matrix(y_test, y_test_pred_best))

print("Classification report (tuned):")
print(classification_report(y_test, y_test_pred_best))





from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix for tuned model
cm_mendely = confusion_matrix(y_test, y_test_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_mendely,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=["Not Impulsive", "Impulsive"],
    yticklabels=["Not Impulsive", "Impulsive"]
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix – KNN (Mendely Dataset)")

# Save high-res image for slides
plt.tight_layout()
plt.savefig("knn_mendely_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

