from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = 0.1
        
    def predict(self, x):
        return 1 if np.dot(x, self.weights) + self.bias > 0 else -1
    
    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                total_error += abs(error)
                
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
            
            # Early stopping se não houver erros
            if total_error == 0:
                break

    def get_weights(self):
        return self.weights.tolist()

    def get_bias(self):
        return float(self.bias)

# Dicionário global para armazenar os perceptrons treinados
perceptrons = {}

@app.route('/train', methods=['POST'])
def train():
    global perceptrons
    try:
        data = request.json
        matrices = np.array(data['matrices'])
        labels = data['labels']
        
        if not all(labels):
            return jsonify({
                'message': 'Por favor, preencha todas as letras'
            }), 400

        # Achatar as matrizes para ter a forma (n_samples, 100)
        flattened_matrices = matrices.reshape(len(matrices), 100)

        # Criar um perceptron para cada letra única
        unique_letters = list(set(labels))
        perceptrons.clear()  # Limpa os perceptrons anteriores
        
        for letter in unique_letters:
            perceptron = Perceptron(100)
            y = np.array([1 if label == letter else -1 for label in labels])
            perceptron.train(flattened_matrices, y, epochs=200)
            perceptrons[letter] = perceptron  # Armazena o perceptron para a letra

        return jsonify({
            'message': f'Modelo treinado com sucesso! Letras reconhecidas: {", ".join(unique_letters)}'
        })

    except Exception as e:
        return jsonify({
            'message': f'Erro durante o treinamento: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        test_matrix = np.array(data['matrix']).reshape(1, 100)  # Achatar a matriz 10x10 para 1x100

        # Dicionário para armazenar as previsões
        predictions = {}

        for letter, perceptron in perceptrons.items():
            prediction = perceptron.predict(test_matrix[0])
            predictions[letter] = prediction

        # Determina a letra com a maior soma de predições (a mais forte)
        predicted_letter = max(predictions, key=predictions.get)
        
        return jsonify({'letra_predita': predicted_letter})

    except Exception as e:
        return jsonify({
            'message': f'Erro durante a previsão: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
