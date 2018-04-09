import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
pd.options.mode.chained_assignment = None



train_url ='C:\\Users\\Defiance\\.spyder-py3\\train.csv'
test_url = 'C:\\Users\\Defiance\\.spyder-py3\\test.csv'


train = pd.read_csv(train_url)
test= pd.read_csv(test_url)

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"]=="female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Age"] = train["Age"].fillna(train["Age"].median())

target = train["Survived"].values

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"] = test["Embarked"].fillna("S")
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test["Age"] = test["Age"].fillna(test["Age"].median())
test.Fare[152] = test.Fare.median()

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)


test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)


Final1 =np.array(test["PassengerId"].astype(int))
Final2 = np.array(pred_forest)
                  
my_solution = pd.DataFrame({"PassengerId" : Final1 ,"Survived" : Final2})

pd.read_excel('C:\\Users\\Defiance\\.spyder-py3\\Book1.xlsx')
my_solution.to_excel("Book1.xlsx")

#this is a comment

print(my_solution)
print("Press Enter to exit")