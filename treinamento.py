import csv

def ler_exemplos_treinamento(arquivo) -> dict:
    exemplos_treinamento = {}

    with open(arquivo, "r", encoding = "utf-8-sig") as arq:
        reader = csv.reader(arq)

        for row in reader:
            row = [int(x) for x in row]

            entradas = tuple(row[:63])
            saidas_esperadas = row[-7:]

            exemplos_treinamento[entradas] = saidas_esperadas

    return exemplos_treinamento

def imprimir_letra(entradas: tuple):
    for i in range(63):
        if i % 7 == 0:
            print()

        if entradas[i] == 1:
            print("#", end = "")
        else:
            print(" ", end = "")

    print()

def reconhecer_letra(saidas: list) -> str:
    letras = ["A", "B", "C", "D", "E", "J", "K"]

    return letras[saidas.index(max(saidas))]

limpo = ler_exemplos_treinamento("dados/limpo.csv")

ruido = ler_exemplos_treinamento("dados/ruido.csv")
