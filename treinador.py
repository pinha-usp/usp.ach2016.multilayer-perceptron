import json
import jsbeautifier
from copy import deepcopy

from exemplos import exemplos
from mlp import MLP

class Treinador:

    def __init__(
        self,
        camadas: list,
        taxa_aprendizado: float
    ):
        self.mlp = MLP(camadas, taxa_aprendizado)

        # Informações iniciais (antes do treinamento) do MLP copiadas
        self.mlp_inicial = deepcopy(self.mlp)

    def treinar(self, exemplos: list, epocas: int):
        self.mlp.treinar(exemplos, epocas)

    def gerar_arquitetura(self) -> dict:
        return {
            "camadas": [63, 21, 7],
            "taxa_aprendizado": self.mlp.taxa_aprendizado
        }

    def gerar_erros(self) -> dict:
        return [
            round(eqm, 5)
            for eqm 
            in self.mlp.eqms
        ]

    def gerar_pesos(self) -> dict:
        def gerar_pesos_mlp(mlp: MLP) -> list:
            return [
                [
                    [
                        round(peso, 3)
                        for peso
                        in neuronio.pesos
                    ]
                    for neuronio
                    in camada
                ]
                for camada
                in mlp.camadas
            ]

        return {
            "iniciais": gerar_pesos_mlp(self.mlp_inicial),
            "finais": gerar_pesos_mlp(self.mlp)
        }

    def gerar_saidas(self) -> dict:
        def gerar_saidas_exemplos(exemplos: list) -> list:
            return [
                {
                    "esperado": exemplo["letra"],
                    "obtidas": [
                        round(saida, 2)
                        for saida
                        in self.mlp.executar(exemplo["entradas"])
                    ]
                }
                for exemplo
                in exemplos
            ]

        return { 
            "limpos": gerar_saidas_exemplos(exemplos["limpos"]),
            "ruidos": gerar_saidas_exemplos(exemplos["ruidos"]) 
        }

    def salvar_resultados(self):
        alvos = {
            "resultados/arquitetura.json": self.gerar_arquitetura(), 
            "resultados/erros.json": self.gerar_erros(),
            "resultados/pesos.json": self.gerar_pesos(),
            "resultados/saidas.json": self.gerar_saidas()
        }

        for nome_arquivo, resultado in alvos.items():
            with open(nome_arquivo, "w") as arquivo:
                arquivo.write(
                    jsbeautifier.beautify(
                        json.dumps(resultado)
                    )
                )
