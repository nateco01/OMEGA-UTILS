"""Utilitários compartilhados da interface OMEGA Utils.

Inclui um avaliador seguro de expressões matemáticas (sem ``eval``) e
conversões de tipos para os formulários HTML.
"""

from __future__ import annotations

import ast
import math
import re


# ---------------------------------------------------------------------------
# Avaliador seguro de expressões
# ---------------------------------------------------------------------------

_CONSTANTES = {
    "pi": math.pi,
    "PI": math.pi,
    "e": math.e,
    "E": math.e,
    "phi": (1 + math.sqrt(5)) / 2,
    "tau": 2 * math.pi,
    "inf": math.inf,
}

_FUNCOES = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "cbrt": lambda v: math.copysign(abs(v) ** (1 / 3), v),
    "abs": abs,
    "fabs": math.fabs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "min": min,
    "max": max,
    "pow": math.pow,
    "hypot": math.hypot,
    "factorial": math.factorial,
    "gamma": math.gamma,
}

_OPERADORES_BINARIOS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_OPERADORES_UNARIOS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class ErroExpressao(ValueError):
    """Erro amigável para expressões inválidas."""


def _avaliar_no(node, variaveis):
    if isinstance(node, ast.Expression):
        return _avaliar_no(node.body, variaveis)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ErroExpressao(f"valor não numérico: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in variaveis and variaveis[node.id] is not None:
            return variaveis[node.id]
        if node.id in _CONSTANTES:
            return _CONSTANTES[node.id]
        raise ErroExpressao(f"variável desconhecida: '{node.id}'")
    if isinstance(node, ast.BinOp):
        opcao = _OPERADORES_BINARIOS.get(type(node.op))
        if opcao is None:
            raise ErroExpressao("operador não permitido")
        return opcao(_avaliar_no(node.left, variaveis), _avaliar_no(node.right, variaveis))
    if isinstance(node, ast.UnaryOp):
        opcao = _OPERADORES_UNARIOS.get(type(node.op))
        if opcao is None:
            raise ErroExpressao("operador unário não permitido")
        return opcao(_avaliar_no(node.operand, variaveis))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ErroExpressao("chamada inválida")
        nome = node.func.id
        funcao = _FUNCOES.get(nome)
        if funcao is None:
            raise ErroExpressao(f"função desconhecida: '{nome}'")
        argumentos = [_avaliar_no(a, variaveis) for a in node.args]
        try:
            return funcao(*argumentos)
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ErroExpressao(f"erro em {nome}(...): {exc}") from exc
    raise ErroExpressao("construção não permitida na expressão")


def compilar_expressao(texto, variaveis=("x", "t")):
    """Compila uma expressão segura; retorna uma função f(**kwargs)."""
    texto_limpo = (texto or "").strip()
    if not texto_limpo:
        raise ErroExpressao("expressão vazia")
    # Notação implícita simples: 2x -> 2*x, 3sin(x) -> 3*sin(x)
    texto_limpo = re.sub(r"(\d)\s*(?=[A-Za-z(])", r"\1*", texto_limpo)
    try:
        arvore = ast.parse(texto_limpo, mode="eval")
    except SyntaxError as exc:
        raise ErroExpressao(f"sintaxe inválida: {exc.msg}") from exc

    def avaliar(**valores):
        variaveis_ok = {nome: valores.get(nome) for nome in variaveis}
        resultado = _avaliar_no(arvore, variaveis_ok)
        if isinstance(resultado, complex):
            raise ErroExpressao("resultado complexo não suportado")
        return float(resultado)

    return avaliar


def criar_funcao(texto, variaveis=("x", "t")):
    """Retorna (funcao, expressao_formatada) ou lança ErroExpressao."""
    func = compilar_expressao(texto, variaveis)
    return func, texto.strip()


# ---------------------------------------------------------------------------
# Conversões de formulário
# ---------------------------------------------------------------------------

def numero(valor, nome="valor", permitir_vazio=None):
    """Converte string do formulário em float, com erro amigável."""
    if valor is None or (isinstance(valor, str) and not valor.strip()):
        if permitir_vazio is not None:
            return permitir_vazio
        raise ErroExpressao(f"informe o campo '{nome}'.")
    try:
        resultado = float(valor)
    except (TypeError, ValueError):
        raise ErroExpressao(f"'{nome}' deve ser um número.") from None
    if not math.isfinite(resultado) and abs(resultado) != math.inf:
        raise ErroExpressao(f"'{nome}' deve ser finito.")
    return resultado


def matriz(texto, nome="matriz"):
    """Converte texto '[1,2;3,4]' ou '[[1,2],[3,4]]' em lista de listas."""
    dados = (texto or "").strip()
    if not dados:
        raise ErroExpressao(f"informe a '{nome}'.")
    dados = dados.replace(";", ",").replace("\n", ",")
    dados = re.sub(r",\s*,", ",", dados).strip(", ")
    if not dados.startswith("["):
        dados = "[" + dados + "]"
    try:
        arrastado = ast.literal_eval(dados)
    except (SyntaxError, ValueError):
        raise ErroExpressao(
            f"'{nome}' inválida. Use colchetes, ex.: [[1,2],[3,4]] ou [1,2;3,4]."
        ) from None

    def normalizar(item):
        if isinstance(item, (int, float)):
            return [float(item)]
        if isinstance(item, list):
            linhas = [normalizar(subitem) for subitem in item]
            if any(len(linha) != len(linhas[0]) for linha in linhas):
                if all(len(linha) == 1 for linha in linhas):
                    return [linha[0] for linha in linhas]
                raise ErroExpressao(f"linhas da '{nome}' com tamanhos diferentes.")
            if all(len(linha) == 1 for linha in linhas) and all(
                isinstance(subitem, (int, float)) for subitem in item
            ):
                return [linha[0] for linha in linhas]
            return linhas
        raise ErroExpressao(f"'{nome}' contém valores não numéricos.")

    resultado = normalizar(arrastado)
    return resultado


def vetor(texto, nome="vetor"):
    """Converte '[1,2,3]' em lista plana de floats."""
    linhas = matriz(texto, nome)
    plano = []
    for linha in linhas:
        if isinstance(linha, list):
            plano.extend(float(v) for v in linha)
        else:
            plano.append(float(linha))
    return plano


def formatar_numero(valor, digitos=6):
    """Formata números para exibição amigável (notação científica se preciso)."""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "∞" if v > 0 else "-∞"
    if v == 0:
        return "0"
    absoluto = abs(v)
    if absoluto >= 1e8 or absoluto < 1e-4:
        return f"{v:.{digitos}e}"
    texto = f"{v:.{digitos}g}"
    return texto
