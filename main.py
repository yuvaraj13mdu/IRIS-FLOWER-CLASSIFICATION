import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from warnings import filterwarnings
filterwarnings(action='ignore')

iris=pd.read_csv("C:\\Users\\welcme\\Desktop\\CODSOFT\\iris.csv")
print(iris)

print(iris.shape)

print(iris.describe())

#Checking for null values

print(iris.isna().sum())
print(iris.describe())

iris.head()

iris.head(150)

iris.tail(100)

print(iris.columns)

n = len(iris[iris['species'] == 'versicolor'])
print("No of Versicolor in Dataset:",n)

n2 = len(iris[iris['species'] == 'setosa'])
print("No of Setosa in Dataset:",n2)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()

#Checking for outliars

plt.figure(1)
plt.boxplot([iris['sepal_length']])
plt.figure(2)
plt.boxplot([iris['sepal_width']])
plt.show()

iris.hist()
plt.show()

iris.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)

iris.plot(kind ='box',subplots = True, layout =(2,5),sharex = False)

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='petal_length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='petal_width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='sepal_width',data=iris)

sns.pairplot(iris,hue='species');

# Select only numeric columns for correlation calculation
numeric_iris = iris.select_dtypes(include=[float, int])

# Calculate the correlation matrix
corr_matrix = numeric_iris.corr()

# Create a heatmap of the correlation matrix
fig = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=1, linecolor='k', square=True,
                 vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"}, cbar=True)

# Show the plot
plt.show()

X = iris['sepal_length'].values.reshape(-1,1)
print(X)

Y = iris['sepal_width'].values.reshape(-1,1)
print(Y)

plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.scatter(X,Y,color='b')
plt.show()

#Correlation

# Exclude the "Species" column from the correlation calculation
numeric_iris = iris.drop(columns=["species"])

# Calculate the correlation matrix for the numeric columns
corr_matrix = numeric_iris.corr()

# Print the correlation matrix
print(corr_matrix)

train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)

train_X = train[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
train_y = train.species

test_X = test[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
test_y = test.species

train_X.head()

test_y.head()

test_y.head()

#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))

#Confusion matrix

confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))

#Using Support Vector

model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

print("Acc=",accuracy_score(test_y,pred_y))

#Using KNN Neighbors

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

print("Accuracy Score:",accuracy_score(test_y,y_pred2))

#Using GaussianNB

model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

print("Accuracy Score:",accuracy_score(test_y,y_pred3))

#Using Decision Tree
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

print("Accuracy Score:",accuracy_score(test_y,y_pred4))

#RESULTS

results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [0.947,0.947,0.947,0.947,0.921]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)

print(results)
