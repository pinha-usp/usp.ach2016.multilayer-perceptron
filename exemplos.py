import json

tipos_dados = ["limpos", "ruidos"]

exemplos = {
    tipo: json.load(
        open(f"dados/{tipo}.json")
    )
    for tipo
    in tipos_dados
}
