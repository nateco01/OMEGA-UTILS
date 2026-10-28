import json
import os

_Caminho = os.path.dirname(__file__)

class ElementosMateriais:
    """Gerenciador de dados científicos locais para elementos, fluidos e estruturas cristalinas.
    
    Permite consultas aninhadas rápidas e offline em arquivos JSON estruturados.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def carregar_elemento(*args):
        """Carrega dados atômicos e propriedades físico-químicas de elementos da tabela periódica.

        Args:
            *args (str): Chaves hierárquicas para busca no JSON. 
                         Ex: ("H",) para todos os dados do Hidrogênio, 
                         ou ("C", "atomic_weight") para obter apenas a massa atômica.

        Returns:
            Any: O valor, dicionário ou subpropriedade correspondente aos argumentos informados.

        Example:
            >>> ElementosMateriais.carregar_elemento("H", "atomic_weight")
            1.008
        """
        caminho = os.path.join(_Caminho, "elementos_tabela_periodica.json")
        with open(caminho, encoding="utf-8") as f:
            data = json.load(f)
        
        atual = data
        for chave in args:
            atual = atual[chave]
        return atual

    @staticmethod
    def carregar_fluido(*args):
        """Carrega propriedades termodinâmicas e pontos críticos de fluidos e substâncias puras.

        Args:
            *args (str): Chaves hierárquicas para busca no JSON. 
                         Ex: ("Water",) para dados da água, 
                         ou ("Water", "critical_temperature_K") para a temperatura crítica.

        Returns:
            Any: O valor, dicionário ou subpropriedade correspondente aos argumentos informados.

        Example:
            >>> ElementosMateriais.carregar_fluido("Water", "critical_temperature_K")
            647.096
        """
        caminho = os.path.join(_Caminho, "fluidos_termodinamicos.json")
        with open(caminho, encoding="utf-8") as f:
            data = json.load(f)

        atual = data
        for chave in args:
            atual = atual[chave]
        return atual

    @staticmethod
    def carregar_materiais(*args):
        """Carrega dados de estruturas cristalinas, vetores de rede e posições atômicas de sólidos.

        Args:
            *args (str): Chaves hierárquicas para busca no JSON. 
                         Ex: ("Si",) para o silício diamante, 
                         ou ("Si", "lattice_parameters", "a") para o parâmetro de rede.

        Returns:
            Any: O valor, dicionário ou subpropriedade correspondente aos argumentos informados.

        Example:
            >>> ElementosMateriais.carregar_materiais("Si", "crystal_structure")
            'diamond'
        """
        caminho = os.path.join(_Caminho, "estruturas_cristalinas.json")
        with open(caminho, encoding="utf-8") as f:
            data = json.load(f)

        atual = data
        for chave in args:
            atual = atual[chave]

        return atual