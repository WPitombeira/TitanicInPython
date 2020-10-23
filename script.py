import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Challenge: Titanic
## Objective: Predict which passengers survived and died in Titanic catastrophe

path = 'C:/Users/user/Documents/dev/python/challenges/titanic'

train = pd.read_csv(F'{path}/data/train.csv')

#train.describe()

# testar ordinal encoder com strings e numeros juntos
# column transformer
# get_dummys

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

sex_cat = train[["Sex"]]
train['Sexf'] = ordinal_encoder.fit_transform(sex_cat)

cabin = train[["Cabin"]].fillna('U')
train["Cabinf"] = ordinal_encoder.fit_transform(cabin)

emb = train[["Embarked"]].fillna('U')
train['emb'] = ordinal_encoder.fit_transform(emb)

features = ['Fare', 'Cabinf', 'Pclass', 'emb', 'Sexf']

# Inserção da média nos valores NA
train.fillna(train.mean(), inplace=True)


x = train[features]
y = train.Survived

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)

###
# ALGORITHM SELECTION
###

models = []
modelsName = []

from sklearn.ensemble import RandomForestClassifier
models.append(RandomForestClassifier(n_estimators = 1000, max_depth = 10, random_state = 1))
modelsName.append("RF")

################################

from sklearn import svm
models.append(svm.SVC(max_iter = 1000, random_state = 1, kernel = 'sigmoid'))
modelsName.append("SVC")

################################

from sklearn.naive_bayes import GaussianNB
models.append(GaussianNB())
modelsName.append("Gaussian")

################################

from sklearn import tree
models.append(tree.DecisionTreeClassifier(criterion= 'entropy'))
modelsName.append("dTree")

################################

from sklearn.neural_network import MLPClassifier
models.append(MLPClassifier(solver = 'adam', activation='tanh'))
modelsName.append("MLP")

################################

# SGDClassifier took us a result so bad i choose remove it from tests

################################

from sklearn.neighbors import KNeighborsClassifier
models.append(KNeighborsClassifier(weights='distance'))
modelsName.append("KNN")

################################

########### TRAINING MODELS ###############

for model in models:
     model.fit(train_X, train_y)
 
x = []
for model in models:
     x.append(model.score(val_X, val_y))

plt.scatter(modelsName, x, s=60, alpha = 0.5, label = "Performance", c = ["red", "green", "brown", "lime", "black", "blue"] )
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")


###
# TEST
###


test = pd.read_csv(F'{path}/data/test.csv')


sex_cat = test[["Sex"]]
test['Sexf'] = ordinal_encoder.fit_transform(sex_cat)

cabin = test[["Cabin"]].fillna('U')
test["Cabinf"] = ordinal_encoder.fit_transform(cabin)

emb = test[["Embarked"]].fillna('U')
test['emb'] = ordinal_encoder.fit_transform(emb)

test_data = test[features]

test_data["Fare"].fillna(test_data["Fare"].mean())

# Inserção da média nos valores NA
test_data.fillna(test_data.mean(), inplace=True)

test_preds = []

for model in models:
     test_preds.append(model.predict(test_data))

#test_pred = model.predict(test_data)

for pred in test_preds:
     i=1
     output = pd.DataFrame({'PassengerId': test.PassengerId,
                       'Survived': pred})
     output.to_csv(F'{path}/submissions/submission'+str(np.random.rand())+'.csv', index=False)
     i++
