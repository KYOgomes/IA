import numpy as np

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicializar os dados de entrada e saída
def generate_data(func_type, n_inputs):
    inputs = np.array([list(map(int, f"{i:0{n_inputs}b}")) for i in range(2 ** n_inputs)])
    if func_type == "AND":
        outputs = np.array([[np.all(row)] for row in inputs])
    elif func_type == "OR":
        outputs = np.array([[np.any(row)] for row in inputs])
    elif func_type == "XOR":
        outputs = np.array([[np.sum(row) % 2] for row in inputs])
    else:
        raise ValueError("Tipo de função desconhecida. Escolha AND, OR ou XOR.")
    return inputs, outputs

# Rede neural com uma camada oculta
class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.weights_input_hidden = np.random.rand(n_inputs, n_hidden)
        self.weights_hidden_output = np.random.rand(n_hidden, n_outputs)
        self.bias_hidden = np.random.rand(1, n_hidden)
        self.bias_output = np.random.rand(1, n_outputs)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = sigmoid(final_input)

            # Cálculo do erro
            error = y - final_output
            if epoch % 1000 == 0:
                print(f"Época {epoch}, Erro: {np.mean(np.abs(error))}")

            # Backpropagation
            output_gradient = error * sigmoid_derivative(final_output)
            hidden_gradient = np.dot(output_gradient, self.weights_hidden_output.T) * sigmoid_derivative(hidden_output)

            # Atualização de pesos e biases
            self.weights_hidden_output += np.dot(hidden_output.T, output_gradient) * learning_rate
            self.weights_input_hidden += np.dot(X.T, hidden_gradient) * learning_rate
            self.bias_output += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate
            self.bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = sigmoid(final_input)
        return (final_output > 0.5).astype(int)

# Main
if __name__ == "__main__":
    print("Escolha a função lógica (AND, OR, XOR): ")
    func_type = input().strip().upper()

    print("Digite o número de entradas booleanas (ex.: 2 ou 10): ")
    n_inputs = int(input())

    X, y = generate_data(func_type, n_inputs)

    print(f"Treinando para {func_type} com {n_inputs} entradas...")
    nn = NeuralNetwork(n_inputs=n_inputs, n_hidden=5, n_outputs=1)
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    print("Resultados:")
    predictions = nn.predict(X)
    for input_val, prediction, target in zip(X, predictions, y):
        print(f"Entrada: {input_val}, Saída esperada: {target}, Previsão: {prediction}")
