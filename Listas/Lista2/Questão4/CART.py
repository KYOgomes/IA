from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Carregando o dataset Iris como exemplo
data = load_iris()
X = data.data  
y = data.target  

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo CART (Árvore de Decisão)
clf = DecisionTreeClassifier(criterion='gini', random_state=42)

# Treinando o modelo
clf.fit(X_train, y_train)

# Avaliando o modelo
accuracy = clf.score(X_test, y_test)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

# Visualizando a árvore de decisão
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()
