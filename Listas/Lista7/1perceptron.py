import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(n_inputs + 1)  # Inclui o bias como peso extra
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
                errors += abs(error)
            self.plot_decision_boundary(training_inputs, labels, epoch)
            if errors == 0:
                print(f"Treinamento finalizado na época {epoch + 1}")
                break

    def plot_decision_boundary(self, inputs, labels, epoch):
        if inputs.shape[1] == 2:  # Só podemos plotar para 2D
            x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
            y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = -(self.weights[1] * x_vals + self.weights[0]) / self.weights[2]

            plt.figure(figsize=(8, 6))
            plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap=plt.cm.Paired)
            plt.plot(x_vals, y_vals, 'r--', label=f'Época {epoch + 1}')
            plt.title(f'Decisão - Época {epoch + 1}')
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.legend()
            plt.show()


# Função para treinar e testar o Perceptron
def testar_perceptron(funcao_logica, n_entradas):
    # Gerar os dados de treinamento para AND, OR e XOR
    entradas = np.array([list(map(int, f"{i:0{n_entradas}b}")) for i in range(2 ** n_entradas)])
    if funcao_logica == "AND":
        saidas = np.all(entradas, axis=1).astype(int)
    elif funcao_logica == "OR":
        saidas = np.any(entradas, axis=1).astype(int)
    elif funcao_logica == "XOR":
        saidas = np.sum(entradas, axis=1) % 2
    else:
        raise ValueError("Função lógica inválida. Escolha entre 'AND', 'OR', ou 'XOR'.")

    # Criar e treinar o Perceptron
    perceptron = Perceptron(n_inputs=n_entradas)
    perceptron.train(entradas, saidas)

    # Testar o modelo
    print(f"\nResultados para {funcao_logica} com {n_entradas} entradas:")
    for entrada, saida in zip(entradas, saidas):
        predicao = perceptron.predict(entrada)
        print(f"Entrada: {entrada}, Esperado: {saida}, Predito: {predicao}")


# Testes
print("Treinando para AND com 2 entradas:")
testar_perceptron("AND", 2)

print("\nTreinando para OR com 2 entradas:")
testar_perceptron("OR", 2)

print("\nTentando resolver XOR com 2 entradas:")
testar_perceptron("XOR", 2)
