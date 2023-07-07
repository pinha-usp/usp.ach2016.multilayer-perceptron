from math import exp
from random import uniform 

class Metadados:
    """
    Metadados de um neurônio, contendo informações sobre a última execução dele,
    como entradas, soma ponderada sobre as entradas, saída e erro calculado para
    a saída
    """

    def __init__(self):
        self.entradas = None
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
            uniform(-1, 1)
            for _
            in range(num_entradas)
        ]
        self.bias = uniform(-1, 1)
        self.ativacao = ativacao
        self.metadados = Metadados()
        
    def executar(self, entradas: list):
        """
        As entradas do neurônio serão as entradas da rede neural ou as saídas
        da camada anterior à camada do neurônio
        """

        self.metadados.entradas = entradas

        ponderada = sum(
            entrada * peso
            for entrada, peso
            in zip(entradas, self.pesos)
        )

        self.metadados.saida = self.ativacao(
            ponderada + self.bias
        )

        return self.metadados.saida

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

            self.bias = (
                self.bias +
                self.metadados.erro *
                taxa_aprendizado
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

        for num_entradas, num_neuronios in zip(
            camadas[:-1],
            camadas[1:]
        ):
            camada = [
                Neuronio(
                    num_entradas,
                    self.ativacao
                )
                for _
                in range(num_neuronios)
            ]
            self.camadas.append(camada)

        self.taxa_aprendizado = taxa_aprendizado

        # Erro quadrático médio para cada época de um treinamento
        self.eqms = []

    def ativacao(self, ponderada: float):
        """Função de ativação Sigmoide"""

        return -1 + 2 / (1 + exp(-ponderada))
    
    def derivada(self, ponderada: float):
        """
        Derivada da função de ativação Sigmoide. Usada para calcular o erro
        associado a cada neurônio da rede
        """

        return 0.5 * (1 + ponderada) * (1 - ponderada)

    def calcular_eqm(self, todas_esperadas: list, todas_obtidas: list):
        """
        Calcula o erro quadrático médio da rede neural com base em todas as saídas 
        esperadas e obtidas para uma época. Esse erro é utilizado para realizarmos
        a parada antecipada
        """

        eqm = 0

        for esperadas, obtidas in zip(todas_esperadas, todas_obtidas):
            eqm += sum(
                (esperada - obtida) ** 2
                for esperada, obtida
                in zip(esperadas, obtidas)
            )

        eqm = eqm / len(todas_esperadas)

        return eqm

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
                neuronio.executar(e)
                for neuronio
                in camada
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
            erro *= self.derivada(neuronio.metadados.saida)

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
                erro *= self.derivada(neu_o.metadados.saida)

                neu_o.errou(erro)

    def atualizar_pesos(self):
        """
        Atualiza todos os pesos da rede neural com base nos erros calculados
        para cada neurônio
        """

        for _, camada in enumerate(self.camadas):
            for neuronio in camada:
                neuronio.atualizar_pesos(self.taxa_aprendizado)

    def treinar(self, exemplos: dict, epocas: int):
        """
        O treinamento consiste em executar o feedforward, calcular os erros da
        camada de saída, propagar os erros para as camadas ocultas e atualizar
        os pesos de todos os neurônios da rede neural. Esse processo é repetido
        por um número determinado de épocas

        Cada exemplo de treinamento é um par de entradas e saídas esperadas
        """

        print("Inicializando treinamento para", epocas, "épocas...")

        for e in range(epocas):
            todas_esperadas = []
            todas_obtidas = []

            for exemplo in exemplos:
                entradas = exemplo["entradas"]
                saidas_esperadas = exemplo["saidas_esperadas"]
                saidas_obtidas = self.executar(entradas)

                todas_esperadas.append(saidas_esperadas)
                todas_obtidas.append(saidas_obtidas)

                self.calcular_erros_saida(saidas_esperadas)
                self.backpropagation()
                self.atualizar_pesos()

            eqm = self.calcular_eqm(todas_esperadas, todas_obtidas)

            print("Época:", e, "EQM:", eqm)

            self.eqms.append(eqm)

    def executar(self, entradas: list):
        """Executa as entradas na MLP e retorna as saídas obtidas"""

        self.feedforward(entradas)

        camada_saida = self.camadas[-1]

        return [
            neuronio.metadados.saida
            for neuronio
            in camada_saida
        ]
