import yaml

def carregar_configuracao(caminho_arquivo):
    with open(caminho_arquivo, 'r') as arquivo:
        configuracao = yaml.safe_load(arquivo)
    return configuracao
