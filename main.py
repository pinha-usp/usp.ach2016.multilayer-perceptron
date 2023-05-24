from mlp import MLP
from treinamento import (
    imprimir_letra,
    reconhecer_letra,
    limpo,
    ruido,
)

mlp = MLP(
    camadas = [63, 21, 7],
    taxa_aprendizado = 0.1
)

mlp.treinar(
    limpo,
    epocas = 8000
)

for letra in ruido.keys():
    saidas = mlp.executar(letra)

    imprimir_letra(letra)
    print(f"> {saidas}")
    print(reconhecer_letra(saidas))
