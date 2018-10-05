import pydotplus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.externals.six import StringIO 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Carrega dataset com Pandas
df = pd.read_csv('pulsar_stars.csv')

#Esse dataset eh um classificador de Pulsar Stars. Onde 1 = pulsar star | 0 = not pulsar star

#Numero de pulsar stars identificado como 1639
#print (sum(df['target_class']))

#Gera um histograma do dataset bruto
df.hist(grid=False)

#Normaliza o dataset
normalizado = (df-df.min())/(df.max()-df.min())

#Gera um histograma do dataset normalizado
normalizado.hist(grid=False, color='skyblue')

#Para balanceamento o dataset, pegamos 1639 not stars (undersampling)
not_stars = df[df['target_class'] == 0].sample(1639)

#Pega as pulsar stars
stars = df[df['target_class'] == 1].sample(1639)

# #Concatena ambas formando um dataset balanceado
X = pd.concat([not_stars.drop(['target_class'], axis=1), stars.drop(['target_class'], axis=1)])
y = pd.concat([not_stars[['target_class']], stars[['target_class']]])

y.hist(grid=False)

#Declara classificadores
decision_tree = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier(3)

#Declara listas
knn_preds = []
tree_preds = []
test_size = []

for i in np.arange(0.2, 0.6, 0.1):

    #Divide em train e validation com .8 train e .2 test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    
    #Treina knn
    knn.fit(X_train.values, y_train.values.ravel())

    #Treina decision_tree
    decision_tree.fit(X_train.values, y_train.values.ravel())

    #Salva PDF da arvore gerada (requer pydotplus e graphviz)
    # dot_data = StringIO() 
    # tree.export_graphviz(decision_tree, 
    #                      out_file=dot_data,
    #                      feature_names=list(X_train),
    #                      class_names=['pulsar_star', 'not_star'],
    #                      filled=True, rounded=True,
    #                      impurity=False) 
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    # graph.write_pdf("tree.pdf") 

    #Predictions knn
    predictions_knn = knn.predict(X_test.values)

    #Predictions decision tree
    predictions_tree = decision_tree.predict(X_test.values)

    #Append das acuracias
    knn_preds.append(accuracy_score(y_test.values, predictions_knn))
    tree_preds.append(accuracy_score(y_test.values, predictions_tree))
    test_size.append(i)

plt.plot(test_size, knn_preds)
plt.plot(test_size, tree_preds)
plt.ylabel("Accuracy")
plt.xlabel("Test Size")
plt.legend(["KNN", "Decision Tree"])
plt.axis([0.2, 0.5, 0.5, 1])
plt.show()