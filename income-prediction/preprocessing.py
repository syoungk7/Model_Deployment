import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv('./dataset/sample.csv')

# replace missing values
c_names = df.columns
for c in c_names:
    df[c] = df[c].replace("?", np.NaN)
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

# change column names to simple
df.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married','not married', 'not married', 'not married'], inplace=True)

# drop redundant columns
df = df.drop(['fnlwgt', 'educational-num'], axis=1)


# label Encoder
categories = ['workclass', 'race', 'education','marital-status', 'occupation','relationship', 'gender', 'native-country', 'income']
labelEncoder = preprocessing.LabelEncoder()

# map numerical values to categorical labels
mapping_dict = {}
for col in categories:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col] = le_name_mapping
# print(mapping_dict)

# final featured data and labels
X = df.values[:, 0:12]
Y = df.values[:,12]

# split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# apply model: DecisionTree
dt_clf = DecisionTreeClassifier(criterion = "gini", random_state=100, max_depth=5, min_samples_leaf=5)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print ("Index\nAccuracy using Desicion Tree is ", accuracy_score(y_test, y_pred) * 100)

#creat model as model.pkl
import pickle
pickle.dump(dt_clf, open("model.pkl","wb"))