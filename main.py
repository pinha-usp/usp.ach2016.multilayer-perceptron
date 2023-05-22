from mlp import MLP

perceptron = MLP(camadas = [20, 7], taxa_aprendizado = 0.1)

exemplos_treinamento = {
    [0, 0, 0, 0]: [1, 2, 3, 4]
}

perceptron.treinar(
    exemplos_treinamento,
    epocas = 10
)
