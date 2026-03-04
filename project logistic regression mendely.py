import pandas as pd
mendely=pd.read_csv('/Users/candylam/Desktop/MISM 3515/Raw Data_Paylater and Non Paylater User.csv', sep=';')
mendely.columns
## we take the mean of the variables gives one representative value for that construct per person 
## and reduces noise from random answering differences, keeps everything on the same 1-5 scale,
## and prevents multicollinearity (when many correlated features confuse linear models
## so basically --> combine multiple Likert items into one reliable, interpretable construct 

## we use axis=1 to tell python to operate across columns in the same row , or average across each row

mendely['SelfControl'] = mendely[['SC1', 'SC2','SC3 ', 'SC4 ', 'SC5']].mean(axis=1)
mendely['Happiness'] = mendely[['H1', 'H2', 'H3', 'H4']].mean(axis=1)
mendely['SocialInfluence'] = mendely[['SI1', 'SI2','SI3', 'SI4', 'SI5', 'SI6']].mean(axis=1)
mendely['Promotion'] = mendely[['P1', 'P2', 'P3', 'P4']].mean(axis=1)
mendely['NormalEval'] = mendely[['NE1', 'NE2', 'NE3', 'NE4', 'NE5']].mean(axis=1)
mendely['IBB_Score'] = mendely[['IBB1','IBB2','IBB3','IBB4']].mean(axis=1)

## creating the y (target) variable

mendely['Impulsive_Buyer'] = (mendely['IBB_Score'] >= 3.5).astype(int)
## we're making our threshold here at 3.5 for what is an "impulsive buyer"
##  by the Likert Scale, 3 is neautral, 4-5 is "agree" to "strongly agree"
##  using 3.5 as our cut off means:
    # average < 3.5 --> low impulsivity (generally disagree)
    # average > 3.5 --> high impulsivity (generally agree)
#3 we use .astype(int) because  >= 3.5 returns get a boolean; we're just making sure we're getting an integer back 

mendely['Impulsive_Buyer'].value_counts(normalize=True)
#Impulsive_Buyer
#0    0.722222
#1    0.277778
likert_items_mendely = ['IBB1', 'IBB2', 'IBB3', 'IBB4', 'P1', 'P2', 'P3', 'P4', 'SI1', 'SI2',
'SI3', 'SI4', 'SI5', 'SI6', 'H1', 'H2', 'H3', 'H4', 'SC1', 'SC2',
'SC3 ', 'SC4 ', 'SC5', 'NE1', 'NE2', 'NE3', 'NE4', 'NE5']


mendely = mendely.drop(columns=['IBB1', 'IBB2', 'IBB3', 'IBB4', 'P1', 'P2', 'P3', 'P4', 'SI1', 'SI2',
                                'SI3', 'SI4', 'SI5', 'SI6', 'H1', 'H2', 'H3', 'H4', 'SC1', 'SC2',
                                'SC3 ', 'SC4 ', 'SC5', 'NE1', 'NE2', 'NE3', 'NE4', 'NE5'])

mendely = pd.get_dummies(mendely, drop_first=True)

x_mendely = mendely.drop(columns=['IBB_Score', 'Impulsive_Buyer'])
## we dropped IBB_Score here because we used it to define Impulsive_Buyer; not doing so would result in data leakage

y_mendely=mendely['Impulsive_Buyer']

#Logistic Regression 
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x_mendely,y_mendely,test_size=0.3,
                                                 random_state=1)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(solver='liblinear')
log.fit(x_train, y_train)

log.coef_
log.intercept_ #0.0029568
y_pred=log.predict(x_test)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
f1_score(y_test,y_pred) # 0.5178571428571429
accuracy_score(y_test,y_pred)#0.7777777777777778
precision_score(y_test,y_pred) #0.6590909090909091
recall_score(y_test,y_pred) #0.4264705882352941
y_train_pred=log.predict(x_train)
f1_score(y_train,y_train_pred) #0.5747126436781609

#F1 testing > F1 training, but pretty similar so good balance!
probabilities=log.predict_proba(x_test)
y_prob=probabilities[:,1] ##it extracts class 0 instead of class 1????


from sklearn.metrics import confusion_matrix
cmtxfull=pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]),
             index=['Actual:0','Actual:1'],
             columns=['Pred:0','Pred:1'])
#         Pred:0    Pred:1
#Actual:0     160     15
#Actual:1     39      29
import matplotlib as plt

cmfull=confusion_matrix(y_test,y_pred)

from sklearn.metrics import ConfusionMatrixDisplay
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(cmfull, display_labels=['Non-Impulsive', 'Impulsive']).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

#regression with feature selection
len(x_mendely.columns) #12
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs=SFS(log,
        k_features=(1,12),
        forward=True,
        scoring='f1',
        cv=5)

sfs.fit(x_train,y_train)
sfs.k_feature_names_
#'Monthly Income', 'SelfControl', 'SocialInfluence', 'Promotion'

x_train_sfs=sfs.transform(x_train)
x_test_sfs=sfs.transform(x_test)

log.fit(x_train_sfs,y_train)
y_pred=log.predict(x_test_sfs)

f1=f1_score(y_test,y_pred)
print('f1',f1)
#f1 0.5185185185185185
accuracy_score(y_test,y_pred)#0.7860082304526749
precision_score(y_test,y_pred) #0.7
recall_score(y_test,y_pred) #0.4117647058823529

y_pred_train=log.predict(x_train_sfs)
f1=f1_score(y_train,y_pred_train)
print('f1',f1)
#f1 0.544

#F1 test<F1 train #similar --> overfitting?
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]),
             index=['Actual:0','Actual:1'],
             columns=['Pred:0','Pred:1'])

#          Pred:0  Pred:1
#Actual:0     163    12
#Actual:1     40     28

cmfull=confusion_matrix(y_test,y_pred)

from sklearn.metrics import ConfusionMatrixDisplay
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(cmfull, display_labels=['Non-Impulsive', 'Impulsive']).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

len(x_mendely.columns)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs=SFS(log,
        k_features=(1,12),
        forward=False,
        scoring='f1',
        cv=5)

sfs.fit(x_train,y_train)
sfs.k_feature_names_
#'E-Paylater User Status',
# 'Educational Background',
# 'Year of Birth',
# 'Monthly Income',
# 'Average monthly expenditure for online shopping in relation to monthly income',
# 'SelfControl',
# 'Happiness',
# 'SocialInfluence',
# 'Promotion'

x_train_sfs=sfs.transform(x_train)
x_test_sfs=sfs.transform(x_test)

log.fit(x_train_sfs,y_train)
y_pred=log.predict(x_test_sfs)

f1=f1_score(y_test,y_pred)
print('f1',f1)
#f1 0.5486725663716814
accuracy_score(y_test,y_pred)#0.7901234567901234
precision_score(y_test,y_pred) # 0.6888888888888889
recall_score(y_test,y_pred) #0.45588235294117646

y_pred_train=log.predict(x_train_sfs)
f1=f1_score(y_train,y_pred_train)
print('f1',f1)
#f1 0.5747126436781609

#F1 test<F1 train #similar --> overfitting
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]),
             index=['Actual:0','Actual:1'],
             columns=['Pred:0','Pred:1'])

#          Pred:0  Pred:1
#Actual:0     161    14
#Actual:1     37     31

cmfull=confusion_matrix(y_test,y_pred)

from sklearn.metrics import ConfusionMatrixDisplay
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(cmfull, display_labels=['Non-Impulsive', 'Impulsive']).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()