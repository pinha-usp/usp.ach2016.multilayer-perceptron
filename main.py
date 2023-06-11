from treinador import Treinador
from exemplos import exemplos

treinador = Treinador(
    camadas = [63, 21, 7],
    taxa_aprendizado = 0.1
)

treinador.treinar(
    exemplos["limpos"],
    epocas = 10 
)

treinador.salvar_resultados()
