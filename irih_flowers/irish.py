import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('irish.txt')
x=dataset.iloc[:,:3].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)

# classifiers 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import  cross_val_score, cross_val_predict

cnames = [
            'logistic',
            'kneighbours',
            'svc',
            'decisiontree',
            'randomforest',
            'adaboost',
            'gaussiannb',
            'quadraticdiscriminationanalysis',
        ]

cfiers = [
            LogisticRegression(multi_class='auto',solver='lbfgs'),
            KNeighborsClassifier(),
            SVC(gamma='auto'),
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=100),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()
        ]


baccu=0
bcfier='letssee'

for i in range(len(cnames)):
    print(cnames[i])
    accs = cross_val_score(cfiers[i],x,y,cv=15)
    print(accs.mean(),accs.std())
    if accs.mean()>baccu:
        baccu=accs.mean()
        bcfier=cnames[i]
    print()
    
print('best classifier for the dataset')
print(baccu,bcfier)
    
    
