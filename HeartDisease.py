# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:25:53 2022

@author: msi
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

df=pd.read_csv(r'D:\ebook\semester 9(cse347,cse360,cse366,cse407)\cse366\project\heart\heart_2020_cleaned.csv')

print(df.info())
print(df)
#%%
print(df.head())
#%%
print(df.tail())
#%%
print(df.describe())
#%%
print(df.describe(include=['O']))
#%%
f, ax = plt.subplots(figsize=(5,5))
sns.countplot(x='HeartDisease',data=df)

#%%
data = df['HeartDisease'].value_counts()
labels = ['NO', 'YES',]
colors = sns.color_palette('pastel')[0:5]
plt.pie(data,labels = labels, colors = colors, autopct='%.0f%%')
plt.title('HeartDisease')
plt.show()
#%%
plt.figure(figsize=(15,8))
sns.countplot(x='AgeCategory',data=df)
plt.show()

#%%
'''
plt.figure(figsize=(15,8))
sns.countplot(x='MentalHealth',data=df)
plt.title('MentalHealth')
plt.show()'''
#%%
plt.figure(figsize=(10,4))
sns.boxplot(x='SleepTime',y='HeartDisease', data=df)
#%%
sns.distplot(df['MentalHealth'])
#%%
sns.distplot(df['PhysicalHealth'])
#%%
plt.figure(figsize=(15,8))
sns.countplot(df['GenHealth'])
plt.show()
#%%
data = df['GenHealth'].value_counts()
labels = ['Very good', 'Good','Excellent','Fair','Poor']
colors = sns.color_palette('pastel')[0:5]
plt.pie(data,labels = labels, colors = colors, autopct='%.0f%%')
plt.title('GenHealth')
plt.show()

#%%
data = df['Diabetic'].value_counts()
labels = ['NO', 'YES','No, (borderline diabetes)','Yes (during pregnancy)  ']
colors = sns.color_palette('pastel')[0:5]
plt.pie(data,labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Diabetic')
plt.show()

#%%
sns.boxenplot(x='BMI',data=df)

#%%
plt.figure(figsize=(15,8))
sns.countplot(x='SleepTime',data=df)
plt.title('SleepTime')
plt.show()
#%%
'''
plt.figure(figsize=(15,8))
sns.countplot(x='PhysicalHealth',data=df)
plt.title('PhysicalHealth')
plt.show()'''
#%%
df_0 = df[(df['HeartDisease']=='No')] 
df_1 = df[(df['HeartDisease']=='Yes')] 

#%%
g = sns.FacetGrid(df, col='HeartDisease', height=5, aspect=1.6)
g.map(plt.hist, 'Stroke', alpha=.75, bins=3)
#%%
data = df['Stroke'].value_counts()
labels = ['No','Yes']
colors = sns.color_palette('pastel')[0:5]
plt.pie(data,labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Stroke')
plt.show()

#%%
g = sns.FacetGrid(df_1, col='HeartDisease', height=5, aspect=1)
g.map(plt.hist, 'Smoking', alpha=.75, bins=3)
#%%
g = sns.FacetGrid(df_1, col='HeartDisease', height=5, aspect=1)
g.map(plt.hist, 'AlcoholDrinking', alpha=.75, bins=3)
#%%
g = sns.FacetGrid(df_1, col='HeartDisease', height=5, aspect=1)
g.map(plt.hist, 'Sex', alpha=.75, bins=3)
#%%
g = sns.FacetGrid(df_1, col='HeartDisease', height=5, aspect=1.6)
g.map(plt.hist, 'GenHealth', alpha=.75, bins=9)

#%%
g = sns.FacetGrid(df_1, col='HeartDisease', height=5, aspect=1.6)
g.map(plt.hist, 'MentalHealth', alpha=.75, bins=10)
#%%
g = sns.FacetGrid(df_1, col='HeartDisease', height=5, aspect=2)
g.map(plt.hist, 'Race', alpha=.75, bins=10)

#%%
g = sns.FacetGrid(df_1, col='HeartDisease', height=5, aspect=2)
g.map(plt.hist, 'AgeCategory', alpha=.75, bins=10)

#%%
g = sns.FacetGrid(df_1, col='HeartDisease', height=5, aspect=1)
g.map(plt.hist, 'DiffWalking', alpha=.75, bins=3)
#%%
for column_name in df.columns:
    unique_values = len(df[column_name].unique())
    print("Feature '{column_name}' has '{unique_values}' unique values".format(column_name = column_name,unique_values=unique_values))
    
#%%
df = df.drop(['Race'], axis=1)
print(df.info())
print(type(df))

#%%
cols_yes__no_values = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']



yes__no_values = {'No':0, 'Yes':1}
for i in range(0, len(cols_yes__no_values)):
    df[cols_yes__no_values[i]] = df[cols_yes__no_values[i]].replace(yes__no_values)

sex = {'Female':0, 'Male':1}
df['Sex'] = df['Sex'].replace(sex)

ageCategory= {'18-24':0, '25-29':1, '30-34':2, '35-39':3, '40-44':4, '45-49':5, '50-54':6, '55-59':7, '60-64':8,'65-69':9, '70-74':10, '75-79':11, '80 or older':12}
df['AgeCategory'] = df['AgeCategory'].replace(ageCategory)

genHealth = {'Poor':0, 'Fair':1, 'Good':2, 'Very good':3, 'Excellent':4}
df['GenHealth'] = df['GenHealth'].replace(genHealth)

diabetic = {'No':0, 'No, borderline diabetes':1, 'Yes (during pregnancy)':2, 'Yes':3}
df['Diabetic'] = df['Diabetic'].replace(diabetic)


#%%
print(df)
#%%
print(df.describe())
#%%
plt.figure(figsize=(18,18))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, fmt='.2f')
plt.show()


#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict

x = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

'''
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()    
x= st_x.fit_transform(x)    
'''
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#%%

print()
print()
#%%
'''
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()    
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)  '''
#%%
print("-----------------------------LogisticRegression----------------------------------\n")
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log_train = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log_test = round(logreg.score(X_test, Y_test) * 100, 2)
print('Accuracy of LogisticRegression on training dataset: ',acc_log_train)
print('Accuracy of LogisticRegression on testing dataset: ',acc_log_test)
print()
#%%

#%%
'''
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

x=accuracy_score(Y_test, Y_pred)
print(x)

p_log_test=round(precision_score(Y_test, Y_pred) * 100, 2)
r_log_test=round(recall_score(Y_test, Y_pred) * 100, 2)
f1_log_test=round(f1_score(Y_test, Y_pred) * 100, 2)

print('Precision of LogisticRegression on testing dataset: ',p_log_test)
print('Recall of LogisticRegression on testing dataset: ',r_log_test)
print('F1 score of LogisticRegression on testing dataset: ',f1_log_test)
print()'''
#%%
acc_val_score_train=cross_val_score(logreg, X_train, Y_train, cv=10, scoring="accuracy")
acc_val_score_test=cross_val_score(logreg, X_test, Y_test, cv=10, scoring="accuracy")

acc_val_mean_train_logreg=round(acc_val_score_train.mean()*100,2)
acc_val_min_train_logreg=round(acc_val_score_train.min()*100,2)
acc_val_max_train_logreg=round(acc_val_score_train.max()*100,2)

acc_val_mean_test_logreg=round(acc_val_score_test.mean()*100,2)
acc_val_min_test_logreg=round(acc_val_score_test.min()*100,2)
acc_val_max_test_logreg=round(acc_val_score_test.max()*100,2)

print("Accuracy after cross val of LogisticRegression on training dataset: ",acc_val_mean_train_logreg)
print(f"where min score: {acc_val_min_train_logreg}  \n  and max score: {acc_val_max_train_logreg}")
print()
print("Accuracy after cross val of LogisticRegression on testing dataset: ",acc_val_mean_test_logreg)
print(f"where min score: {acc_val_min_test_logreg}  \n  and max score: {acc_val_max_test_logreg}")
print()
print()
#%%
#Visualizing the training set result  
'''
from matplotlib.colors import ListedColormap  
x_set, y_set = Y_pred, Y_test  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, logreg.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
plt.title('Logistic Regression (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show() '''
#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

CM_GB = confusion_matrix(Y_test, Y_pred)
ax = sns.heatmap(CM_GB/np.sum(CM_GB),fmt='.2%', annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
ax.yaxis.set_ticklabels(['No HeartDisease','HeartDisease'])

plt.show()
#sns.heatmap(CM_GB, center=True)
#plt.show()

#print('Confusion Matrix is\n', CM_GB)

#score = accuracy_score(Y_test, Y_pred)
#print("Accuracy ", score)

#%%
print("-----------------------------LinearRegression---------------------------------\n")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

linreg = LinearRegression()
linreg.fit(X_train, Y_train)
Y_pred = linreg.predict(X_test)
acc_lin_train = round(linreg.score(X_train, Y_train) * 100, 2)
acc_lin_test = round(linreg.score(X_test, Y_test) * 100, 2)
print('Accuracy of LinearRegression on training dataset: ',acc_lin_train)
print('Accuracy of LinearRegression on testing dataset: ',acc_lin_test)
print()
print()
#%%
sns.regplot(Y_pred, Y_test,line_kws={'color': 'red'})
#plt.scatter(Y_pred, Y_test)
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()
#%%
sns.regplot(x=Y_pred,y= Y_test, logistic=True, ci=None,line_kws={'color': 'red'})
'''
#%%
acc_val_score_train=cross_val_score(linreg, X_train, Y_train, cv=10, scoring="accuracy").mean()*100
acc_val_score_test=cross_val_score(linreg, X_test, Y_test, cv=10, scoring="accuracy").mean()*100
print("train cross val",acc_val_score_train)
print("test cross val",acc_val_score_test)

'''

#%%
print("-----------------------------Naive_Bayes----------------------------------\n")
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian_train = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian_test = round(gaussian.score(X_test, Y_test) * 100, 2)
print('Accuracy of Naive Bayes on training dataset: ',acc_gaussian_train)
print('Accuracy of Naive Bayes on testing dataset: ',acc_gaussian_test)
print()
#%%
acc_val_score_train=cross_val_score(gaussian, X_train, Y_train, cv=10, scoring="accuracy")
acc_val_score_test=cross_val_score(gaussian, X_test, Y_test, cv=10, scoring="accuracy")

acc_val_mean_train_nb=round(acc_val_score_train.mean()*100,2)
acc_val_min_train_nb=round(acc_val_score_train.min()*100,2)
acc_val_max_train_nb=round(acc_val_score_train.max()*100,2)

acc_val_mean_test_nb=round(acc_val_score_test.mean()*100,2)
acc_val_min_test_nb=round(acc_val_score_test.min()*100,2)
acc_val_max_test_nb=round(acc_val_score_test.max()*100,2)

print("Accuracy after cross val of Naive Bayes on training dataset: ",acc_val_mean_train_nb)
print(f"where min score: {acc_val_min_train_nb}  \n  and max score: {acc_val_max_train_nb}")
print()
print("Accuracy after cross val of Naive Bayes on testing dataset: ",acc_val_mean_test_nb)
print(f"where min score: {acc_val_min_test_nb}  \n  and max score: {acc_val_max_test_nb}")
print()
print()
#%%
CM_GB = confusion_matrix(Y_test, Y_pred)
ax = sns.heatmap(CM_GB/np.sum(CM_GB),fmt='.2%', annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
ax.yaxis.set_ticklabels(['No HeartDisease','HeartDisease'])

plt.show()
#%%
print("-----------------------------DecisionTree----------------------------------\n")
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree_train = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree_test = round(decision_tree.score(X_test, Y_test) * 100, 2)
print('Accuracy of Decision Tree on training dataset: ',acc_decision_tree_train)
print('Accuracy of Decision Tree on testing dataset: ',acc_decision_tree_test)
print()
#%%
acc_val_score_train=cross_val_score(decision_tree, X_train, Y_train, cv=10, scoring="accuracy")
acc_val_score_test=cross_val_score(decision_tree, X_test, Y_test, cv=10, scoring="accuracy")

acc_val_mean_train_dt=round(acc_val_score_train.mean()*100,2)
acc_val_min_train_dt=round(acc_val_score_train.min()*100,2)
acc_val_max_train_dt=round(acc_val_score_train.max()*100,2)

acc_val_mean_test_dt=round(acc_val_score_test.mean()*100,2)
acc_val_min_test_dt=round(acc_val_score_test.min()*100,2)
acc_val_max_test_dt=round(acc_val_score_test.max()*100,2)

print("Accuracy after cross val of Decision Tree on training dataset: ",acc_val_mean_train_dt)
print(f"where min score: {acc_val_min_train_dt}  \n  and max score: {acc_val_max_train_dt}")
print()
print("Accuracy after cross val of Decision Tree on testing dataset: ",acc_val_mean_test_dt)
print(f"where min score: {acc_val_min_test_dt}  \n  and max score: {acc_val_max_test_dt}")
print()
print()
#%%
'''
from sklearn import tree
fig = plt.figure(figsize=(25,20))
tree.plot_tree(decision_tree)
'''
#%%
CM_GB = confusion_matrix(Y_test, Y_pred)
ax = sns.heatmap(CM_GB/np.sum(CM_GB),fmt='.2%', annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
ax.yaxis.set_ticklabels(['No HeartDisease','HeartDisease'])

plt.show()

#%%
print("-----------------------------RandomForest----------------------------------\n")
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(100)
rf.fit(X_train,Y_train)

Y_pred = rf.predict(X_test)

acc_rf_train = round(rf.score(X_train, Y_train) * 100, 2)
acc_rf_test = round(rf.score(X_test,Y_test) * 100, 2)

print('Accuracy of Random Forest on training dataset: ',acc_rf_train)
print('Accuracy of Random Forest on test dataset: ',acc_rf_test)
print()


#%%
acc_val_score_train=cross_val_score(rf, X_train, Y_train, cv=10, scoring="accuracy")
acc_val_score_test=cross_val_score(rf, X_test, Y_test, cv=10, scoring="accuracy")

acc_val_mean_train_rf=round(acc_val_score_train.mean()*100,2)
acc_val_min_train_rf=round(acc_val_score_train.min()*100,2)
acc_val_max_train_rf=round(acc_val_score_train.max()*100,2)

acc_val_mean_test_rf=round(acc_val_score_test.mean()*100,2)
acc_val_min_test_rf=round(acc_val_score_test.min()*100,2)
acc_val_max_test_rf=round(acc_val_score_test.max()*100,2)

print("Accuracy after cross val of Random Forest on training dataset: ",acc_val_mean_train_rf)
print(f"where min score: {acc_val_min_train_rf}  \n  and max score: {acc_val_max_train_rf}")
print()
print("Accuracy after cross val of Random Forest on testing dataset: ",acc_val_mean_test_rf)
print(f"where min score: {acc_val_min_test_rf}  \n  and max score: {acc_val_max_test_rf}")
print()
print()
#%%
CM_GB = confusion_matrix(Y_test, Y_pred)
ax = sns.heatmap(CM_GB/np.sum(CM_GB),fmt='.2%', annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
ax.yaxis.set_ticklabels(['No HeartDisease','HeartDisease'])

plt.show()
#%%
print("------------------------------KNeighbors---------------------------------\n")
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn_train = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn_test = round(knn.score(X_test, Y_test) * 100, 2)

print('Accuracy of KNeighbors on training dataset: ',acc_knn_train)
print('Accuracy of KNeighbors on testing dataset: ',acc_knn_test)
print()
#%%
acc_val_score_train=cross_val_score(knn, X_train, Y_train, cv=10, scoring="accuracy")
acc_val_score_test=cross_val_score(knn, X_test, Y_test, cv=10, scoring="accuracy")

acc_val_mean_train_knn=round(acc_val_score_train.mean()*100,2)
acc_val_min_train_knn=round(acc_val_score_train.min()*100,2)
acc_val_max_train_knn=round(acc_val_score_train.max()*100,2)

acc_val_mean_test_knn=round(acc_val_score_test.mean()*100,2)
acc_val_min_test_knn=round(acc_val_score_test.min()*100,2)
acc_val_max_test_knn=round(acc_val_score_test.max()*100,2)

print("Accuracy after cross val of KNeighbors on training dataset: ",acc_val_mean_train_knn)
print(f"where min score: {acc_val_min_train_knn}  \n  and max score: {acc_val_max_train_knn}")
print()
print("Accuracy after cross val of KNeighbors on testing dataset: ",acc_val_mean_test_knn)
print(f"where min score: {acc_val_min_test_knn}  \n  and max score: {acc_val_max_test_knn}")
print()
print()
#%%
CM_GB = confusion_matrix(Y_test, Y_pred)
ax = sns.heatmap(CM_GB/np.sum(CM_GB),fmt='.2%', annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['No HeartDisease','HeartDisease'])
ax.yaxis.set_ticklabels(['No HeartDisease','HeartDisease'])

plt.show()
#%%
print("Sorting the testing score of all Algorithom after cross validation")
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random forest', 'Naive Bayes', 'Decision Tree', 'Linear Regression','KNeighbors'],
    'Score': [acc_val_mean_test_logreg, acc_val_mean_test_rf, acc_val_mean_test_nb, acc_val_mean_test_dt, acc_lin_test, acc_val_mean_test_knn]})
print(models.sort_values(by='Score', ascending=False))