from treinador import Treinador
from exemplos import exemplos

treinador = Treinador(
    camadas = [63, 21, 7],
    taxa_aprendizado = 0.1,
    fator_parada = 0.001
)

treinador.treinar(exemplos["limpos"])

treinador.salvar_resultados()
