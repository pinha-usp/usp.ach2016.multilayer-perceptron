from math import exp
from random import randrange

class Metadados:
    """
    Metadados de um neurônio, contendo informações sobre a última execução dele,
    como entradas, soma ponderada sobre as entradas, saída e erro calculado para
    a saída
    """

    def __init__(self):
        self.entradas = None
        self.ponderada = None
        self.saida = None
        self.erro = None

class Neuronio:
    """
    O neurônio é a unidade básica de uma rede neural artificial. É nele que
    recebemos as entradas, calculamos a soma ponderada e aplicamos a função de
    ativação para obtermos a saída
    """

    def __init__(
        self,
        num_entradas: int,
        ativacao: callable,
    ):
        self.pesos = [
            randrange(-1, 1)
            for _
            in range(num_entradas)
        ]
        self.ativacao = ativacao
        self.metadados = Metadados()
        
    def executar(self, entradas: list, bias: float):
        """
        As entradas do neurônio serão as entradas da rede neural ou as saídas
        da camada anterior à camada do neurônio
        """

        self.metadados.entradas = entradas

        self.metadados.ponderada = sum(
            entrada * peso
            for entrada, peso
            in zip(entradas, self.pesos)
        )

        self.metadados.saida = self.ativacao(
            self.metadados.ponderada + bias
        )

    def errou(self, erro: float):
        """
        O erro do neurônio é calculado no MLP a partir do contexto dele na rede
        neural. Todo neurônio possui um erro associado após a execução da rede
        """

        self.metadados.erro = erro

    def atualizar_pesos(self, taxa_aprendizado: float):
        """
        Atualiza os pesos do neurônio com base nos metadados dele. Essa atualização
        só deve ser feita após o feedforward e o backpropagation
        """

        for i, (peso, entrada) in enumerate(
            zip(
                self.pesos,
                self.metadados.entradas
            )
        ):
            self.pesos[i] = (
                peso +
                self.metadados.erro *
                taxa_aprendizado *
                entrada
            )

class MLP:
    """
    MLP (Multi-Layer Perceptron) é uma rede neural artificial com uma camada de
    entrada, uma ou mais camadas ocultas, e uma camada de saída
    
    Para essa implementação, ignoramos a existência da camada de entrada, pois os
    neurônios de entrada apenas repassam as entradas para os neurônios da primeira
    camada oculta
    """

    def __init__(self, camadas: list, taxa_aprendizado: float):
        # Camadas ocultas e de saída
        self.camadas = []

        # Bias de cada camada
        self.biases = []

        self.taxa_aprendizado = taxa_aprendizado 

    def ativacao(self, soma_ponderada: float):
        """Função de ativação Sigmoide"""

        return 1 / (1 + exp(-soma_ponderada))
    
    def derivada(self, soma_ponderada: float):
        """
        Derivada da função de ativação Sigmoide. Usada para calcular o erro
        associado a cada neurônio da rede
        """

        return soma_ponderada * (1 - soma_ponderada)

    def feedforward(self, entradas: list):
        """
        Recebe as entradas da rede neural e executa o feedforward, ou seja,
        executa cada neurônio da rede neural camada por camada
        """

        # Entradas e saídas da camada atual
        e = entradas
        s = None

        for camada in self.camadas:
            s = [
                neuronio.executar(e, bias)
                for neuronio, bias
                in zip(camada, self.biases)
            ]
            e = s

    def calcular_erros_saida(self, saidas_esperadas: list):
        """
        Os erros de cada neurônio da camada de saída são calculados com base nas
        saídas esperadas da rede neural e nas saídas obtidas por cada neurônio
        após o feedforward
        """

        camada_saida = self.camadas[-1]

        for neuronio, esperada in zip(camada_saida, saidas_esperadas):
            erro = esperada - neuronio.metadados.saida
            erro *= self.derivada(neuronio.metadados.ponderada)

            neuronio.errou(erro)

    def backpropagation(self):
        """
        Calcula os erros dos neurônios de cada camada oculta com base nos erros
        dos neurônios da camada de saída. Os erros são propagados camada por
        camada até a primeira camada oculta
        """

        for oculta, posterior in zip(
            reversed(self.camadas[:-1]),
            reversed(self.camadas[1:])
        ):
            for i, neu_o in enumerate(oculta):
                erro = sum(
                    neu_p.pesos[i] * neu_p.metadados.erro
                    for neu_p
                    in posterior
                )
                erro *= self.derivada(neu_o.metadados.ponderada)

                neu_o.errou(erro)

    def atualizar_pesos(self):
        """
        Atualiza todos os pesos da rede neural com base nos erros calculados
        para cada neurônio
        """

        for camada in self.camadas:
            for neuronio in camada:
                neuronio.atualizar_pesos(self.taxa_aprendizado)

    def treinar(self, exemplos_treinamento: dict, epocas: int):
        """
        O treinamento consiste em executar o feedforward, calcular os erros da
        camada de saída, propagar os erros para as camadas ocultas e atualizar
        os pesos de todos os neurônios da rede neural. Esse processo é repetido
        por um número determinado de épocas

        Cada exemplo de treinamento é um par de entradas e saídas esperadas
        """

        for _ in epocas:
            for (
                entradas,
                saidas_esperadas
            ) in exemplos_treinamento.items():
                self.feedforward(entradas)
                self.calcular_erros_saida(saidas_esperadas)
                self.backpropagation()
                self.atualizar_pesos()
