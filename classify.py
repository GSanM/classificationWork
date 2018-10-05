import pydotplus
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.externals.six import StringIO 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Carrega dataset com Pandas
df = pd.read_csv('pulsar_stars.csv')

#Esse dataset eh um classificador de Pulsar Stars. Onde 1 = pulsar star | 0 = not pulsar star

#Normaliza o dataset
normalizado = (df-df.min())/(df.max()-df.min())

#Numero de pulsar stars identificado como 1639
#print (sum(df['target_class']))

#Para balanceamento o dataset, pegamos 1639 not stars (undersampling)
not_stars = df[df['target_class'] == 0].sample(1639)

#Pega as pulsar stars
stars = df[df['target_class'] == 1]

#Concatena ambas formando um dataset balanceado
X = pd.concat([not_stars.drop(['target_class'], axis=1), stars.drop(['target_class'], axis=1)])
y = pd.concat([not_stars[['target_class']], stars[['target_class']]])

#Divide em train e validation com .8 train e .2 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#Declara classificadores
decision_tree = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier(3)

#Treina knn
knn.fit(X_train, y_train.values.ravel())

#Treina decision_tree
decision_tree.fit(X_train, y_train.values.ravel())

#Salva PDF da arvore gerada (requer pydotplus e graphviz)
dot_data = StringIO() 
tree.export_graphviz(decision_tree, 
                     out_file=dot_data,
                     feature_names=list(X_train),
                     class_names=['pulsar_star', 'not_star'],
                     filled=True, rounded=True,
                     impurity=False) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("tree.pdf") 

#Predictions knn
predictions_knn = knn.predict(X_test)

#Predictions decision tree
predictions_tree = decision_tree.predict(X_test)

#Print das acuracias
print ("KNN Accuracy: ", accuracy_score(y_test, predictions_knn))
print ("Tree Accuracy: ", accuracy_score(y_test, predictions_tree))