import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Challenge: Titanic
## Objective: Predict which passengers survived and died in Titanic catastrophe


def plot_model(X, Y_train, Y_test, xlabel, title):
     plt.figure()
     plt.plot(X, Y_train, label = "Training Accuracy")
     plt.plot(X, Y_test, label = "Test Accuracy")
     plt.ylabel("Accuracy")
     plt.xlabel(xlabel)
     plt.title(title)
     plt.legend()


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

# RF Best 100 = 0.7937
# RF Best 200 = 0.7982
# RF Best 1000/10 = 0.8340

################################

from sklearn import svm
models.append(svm.SVC(max_iter = 1000, random_state = 1, kernel = 'sigmoid'))
modelsName.append("SVC")
# SVM Best rbf = 0.7309
# SVM Best linear = 0.6950
# SVM Best poly = 0.7219
# SVM Best sigmoid = 0.6412

################################

from sklearn.naive_bayes import GaussianNB
models.append(GaussianNB())
modelsName.append("Gaussian")

# Gaussian Naive Bayes
# STD = 0.8206

################################

from sklearn import tree
models.append(tree.DecisionTreeClassifier(criterion= 'entropy'))
modelsName.append("dTree")

# Tree STD = 0.8026
# Tree Entropy = 0.8116

################################

from sklearn.neural_network import MLPClassifier
models.append(MLPClassifier(solver = 'adam', activation='tanh'))
modelsName.append("MLP")

# MLP STD = 0.7892
# MLP lbfgs = 0.77
# MLP lbfgs + logistic = 0.8206
# MLP lbfgs + identity = 0.7757
# MLP lbfgs + tanh = 0.8071

# MLP std + identity = 0.7668
# MLP std + logistic = 0.7847
# MLP std + tanh = 0.8251

################################

# SGDClassifier took us a result so bad i choose remove it from tests

################################

from sklearn.neighbors import KNeighborsClassifier
models.append(KNeighborsClassifier(weights='distance'))
modelsName.append("KNN")

# KNN STD = 0.7533
# KNN STD + DISTANCE = 0.7757

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
     
#####
     
#fig, axs = plt.subplots(2,3, figsize=(15, 6), facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = .8, wspace=.5)

#axs = axs.ravel()

#for model in models:
#    i=1
#    axs[i].plot(pd.DataFrame(model.score(val_X, val_y)))
#    axs[i].set_title("Modelo "+i)
#    i+=1
     

############################################



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


output = pd.DataFrame({'PassengerId': test.PassengerId,
                       'Survived': test_preds})

output.to_csv('submission.csv', index=False)


     
     