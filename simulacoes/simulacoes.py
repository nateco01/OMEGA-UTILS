"""
Módulo de Simulações e Otimização do OMEGA (métodos de Monte Carlo).

Ideia geral
-----------
Todos os métodos aceitam n funções avaliáveis numericamente (equações, derivadas,
integrais, combinações delas — qualquer callable) e n condições de contorno.
Nada é assumido sobre a forma do domínio: ele é a caixa definida por 'limites'
intersectada com as 'condicoes' que você fornecer.

Funções (*args)
    O primeiro argumento é a função principal; as demais entram em seguida
    (*funcoes_adicionais). Todas são avaliadas nos MESMOS pontos amostrados
    (números aleatórios comuns), então as estatísticas conjuntas são coerentes.
    'combinar' (opcional) recebe os valores de todas as funções num ponto e devolve
    uma única grandeza, por exemplo combinar=lambda u, du, I: u**2 + du**2 - I.

Condições de contorno ('condicoes')
    Uma lista de callables ou de objetos Condicao. Um callable "solto" que retorna
    bool define o domínio (True = admissível); um que retorna número é lido como
    g(x) >= 0. Para ser explícito, use Condicao:

        Condicao(lambda x, y: 1 - x**2 - y**2)                      # g >= 0 (desigualdade)
        Condicao(lambda x, y: x**2 + y**2, "igualdade", valor=1.0,  # equação como contorno
                 tolerancia=1e-3)                                  # |g - valor| <= tol
        Condicao(lambda x, y: x > y, "dominio")                     # predicado booleano

    Equações e derivadas entram como Condicao porque são apenas callables numéricos:
    uma derivada limitada, |u'| <= 1, vira Condicao(lambda x: 1 - abs(du(x))).

Retornos e erros
    Erros de uso devolvem uma string "Sinto muito, ..." (convenção do OMEGA).
    Resultados numéricos vêm em dicionários com erro padrão, contagens de amostras
    descartadas, 'avisos' (também emitidos via warnings) e 'limitacoes'.

Otimizador inteligente (OtimizadorInteligente)
    Ajusta os parâmetros de n cálculos (as mesmas funções acima) para atingir um alvo, num espaço
    de busca descrito por objetos Parametro: números reais/inteiros, escolhas (elementos, métodos),
    cadeias (AMINOACIDOS, DNA, RNA), ordens, vetores e composições. O alvo pode ser minimizar,
    maximizar ou atingir um valor/vetor (ajuste a dados). Veja a docstring do método e de Parametro.
"""
import functools
import inspect
import math
import numbers
import warnings
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np



_EXCECOES_NUMERICAS = (ArithmeticError, ValueError)
_POLITICAS_INVALIDOS = ("ignorar", "erro")
_ITERACOES_BISSECCAO = 60
_Z95 = 1.959963984540054


# ----------------------------------------------------------------------------
# Infraestrutura interna
# ----------------------------------------------------------------------------
class _ErroSimulacao(Exception):
    """Erro de uso/validação, convertido em 'Sinto muito, ...' pelos métodos públicos."""


def _retorna_mensagem_de_erro(metodo):
    @functools.wraps(metodo)
    def interno(*args, **kwargs):
        try:
            return metodo(*args, **kwargs)
        except _ErroSimulacao as erro:
            return f"Sinto muito, {erro}"
    return interno


def _avisar(avisos, mensagem):
    avisos.append(mensagem)
    warnings.warn(mensagem, RuntimeWarning, stacklevel=2)


@dataclass(frozen=True)
class Condicao:
    """
    Condição de contorno / restrição avaliada nas coordenadas do ponto.

    tipo:
        'auto'         -> bool: domínio; número: g(x) >= valor        (padrão)
        'dominio'      -> predicado booleano
        'desigualdade' -> g(x) >= valor
        'igualdade'    -> |g(x) - valor| <= tolerancia
    Valores NaN ou exceções aritméticas (divisão por zero, domínio matemático)
    são tratados como "condição violada".
    """
    funcao: Callable
    tipo: str = "auto"
    valor: float = 0.0
    tolerancia: float = 1e-9
    nome: str = ""

    def __post_init__(self):
        if not callable(self.funcao):
            raise _ErroSimulacao("a condição precisa de uma função chamável.")
        if self.tipo not in ("auto", "dominio", "desigualdade", "igualdade"):
            raise _ErroSimulacao(
                "o tipo da condição deve ser 'auto', 'dominio', 'desigualdade' ou 'igualdade'.")
        if not (isinstance(self.tolerancia, numbers.Real) and self.tolerancia >= 0):
            raise _ErroSimulacao("a tolerância da condição deve ser um número >= 0.")

    def _rotulo(self):
        return self.nome or getattr(self.funcao, "__name__", "condição")

    def avaliar(self, *argumentos):
        """
        Retorna (margem, numerica). A condição é satisfeita se e somente se margem >= 0.
        'numerica' é False quando a margem é só ±1 (predicado booleano), isto é,
        quando não há informação contínua (gradiente) sobre a distância à fronteira.
        """
        try:
            v = self.funcao(*argumentos)
        except _EXCECOES_NUMERICAS:
            return -math.inf, False
        tipo = self.tipo
        if tipo == "auto":
            tipo = "dominio" if isinstance(v, (bool, np.bool_)) else "desigualdade"
        if tipo == "dominio":
            return (1.0 if bool(v) else -1.0), False
        try:
            numero = float(v)
        except (TypeError, ValueError):
            raise _ErroSimulacao(
                f"a condição '{self._rotulo()}' deve retornar bool ou um número real "
                f"(retornou {v!r}).") from None
        if math.isnan(numero):
            return -math.inf, True
        if tipo == "desigualdade":
            return numero - self.valor, True
        return self.tolerancia - abs(numero - self.valor), True


def _normalizar_condicoes(condicoes, permitir_igualdade):
    if condicoes is None:
        return []
    if isinstance(condicoes, Condicao) or callable(condicoes):
        condicoes = [condicoes]
    lista = []
    try:
        iterador = iter(condicoes)
    except TypeError:
        raise _ErroSimulacao("'condicoes' deve ser uma lista de funções ou de objetos Condicao.") from None
    for c in iterador:
        if isinstance(c, Condicao):
            cond = c
        elif callable(c):
            cond = Condicao(c)
        else:
            raise _ErroSimulacao(f"condição inválida ({c!r}): use uma função ou um objeto Condicao.")
        if cond.tipo == "igualdade" and not permitir_igualdade:
            raise _ErroSimulacao(
                "condições de igualdade definem um conjunto de medida nula: amostragem por "
                "rejeição nunca o atinge e uma 'faixa de tolerância' mediria outra grandeza "
                "(volume da faixa, não a integral sobre a superfície). Aqui use 'desigualdade' "
                "ou 'dominio'; igualdades só são aceitas em MonteCarloOtimizacao.")
        lista.append(cond)
    return lista


def _margem_minima(condicoes, argumentos, parar_se_violada=False):
    pior = math.inf
    for c in condicoes:
        margem, _ = c.avaliar(*argumentos)
        if margem < pior:
            pior = margem
            if parar_se_violada and pior < 0:
                return pior
    return pior


def _validar_funcoes(principal, extras):
    funcoes = [principal, *extras]
    for i, f in enumerate(funcoes, start=1):
        if not callable(f):
            raise _ErroSimulacao(
                f"o argumento de função nº {i} não é chamável ({f!r}). Se você passou limites, "
                "num_amostras etc. na frente das funções, passe-os como argumentos nomeados "
                "(ex.: num_amostras=50000).")
    return funcoes


def _validar_inteiro(valor, descricao, minimo):
    if isinstance(valor, bool) or not isinstance(valor, numbers.Integral) or valor < minimo:
        raise _ErroSimulacao(f"{descricao} deve ser um inteiro maior ou igual a {minimo}.")
    return int(valor)


def _validar_real_finito(valor, descricao):
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        raise _ErroSimulacao(f"{descricao} deve ser um número real.") from None
    if not math.isfinite(numero):
        raise _ErroSimulacao(f"{descricao} deve ser finito.")
    return numero


def _validar_politica(invalidos):
    if invalidos not in _POLITICAS_INVALIDOS:
        raise _ErroSimulacao("'invalidos' deve ser 'ignorar' ou 'erro'.")


def _validar_combinar(combinar):
    if combinar is not None and not callable(combinar):
        raise _ErroSimulacao("'combinar' deve ser uma função que recebe o valor de cada função.")


def _validar_limites(limites):
    try:
        pares = [(float(a), float(b)) for a, b in limites]
    except (TypeError, ValueError):
        raise _ErroSimulacao(
            "'limites' deve ser uma lista de pares numéricos (a, b), por exemplo [(0, 1), (-1, 1)].") from None
    if not pares:
        raise _ErroSimulacao("'limites' não pode ser vazio.")
    for a, b in pares:
        if not (math.isfinite(a) and math.isfinite(b)):
            raise _ErroSimulacao(
                "os limites devem ser finitos: a amostragem uniforme não cobre domínios infinitos "
                "(use mudança de variável ou amostragem por importância).")
    return pares


def _como_array(valor):
    try:
        a = np.asarray(valor)
    except (TypeError, ValueError):
        raise _ErroSimulacao(f"uma função retornou um valor que não é numérico ({valor!r}).") from None
    if a.dtype.kind in "biu":
        a = a.astype(float)
    elif a.dtype.kind not in "fc":
        raise _ErroSimulacao(f"uma função retornou um valor que não é numérico ({valor!r}).")
    return a


def _avaliar_funcoes(funcoes, combinar, argumentos, escalar_real=False):
    """
    Avalia todas as funções (e 'combinar') em um ponto. Retorna a lista de arrays
    (funções..., combinado) ou None se alguma avaliação for indefinida/não finita.
    """
    try:
        brutos = [f(*argumentos) for f in funcoes]
        if combinar is not None:
            brutos.append(combinar(*brutos[:len(funcoes)]))
    except _EXCECOES_NUMERICAS:
        return None
    saidas = []
    for v in brutos:
        a = _como_array(v)
        if escalar_real and (a.ndim != 0 or a.dtype.kind != "f"):
            raise _ErroSimulacao(
                "aqui as funções devem retornar escalares reais; para saídas vetoriais ou "
                "complexas passe uma função por componente.")
        if not np.all(np.isfinite(a)):
            return None
        saidas.append(a)
    return saidas


def _saida(a):
    a = np.asarray(a)
    return a.item() if a.ndim == 0 else a


class _Acumulador:
    """Média e soma dos quadrados dos desvios (Welford) para escalares/arrays reais ou complexos."""

    def __init__(self):
        self.n = 0
        self.media = None
        self.m2 = None

    def adicionar(self, x):
        if self.media is None:
            self.media = np.zeros(x.shape, dtype=x.dtype)
            self.m2 = np.zeros(x.shape)
        elif x.shape != self.media.shape:
            raise _ErroSimulacao(
                "uma função retornou formatos (shape) diferentes em amostras distintas.")
        self.n += 1
        delta = x - self.media
        self.media = self.media + delta / self.n
        self.m2 = self.m2 + (delta * np.conj(x - self.media)).real

    def variancia(self):
        if self.n < 2:
            return np.zeros(np.shape(self.media))
        return self.m2 / (self.n - 1)

    def resumo(self):
        var = self.variancia()
        return {
            "media": _saida(self.media),
            "desvio_padrao": _saida(np.sqrt(var)),
            "erro_padrao": _saida(np.sqrt(var / self.n)) if self.n > 1 else None,
            "n": self.n,
        }


def _blocos_de_pontos(rng, inferior, superior, total, tamanho_bloco=8192):
    restante = total
    while restante > 0:
        n = min(tamanho_bloco, restante)
        bloco = rng.uniform(inferior, superior, size=(n, len(inferior)))
        yield from bloco.tolist()
        restante -= n


def _resumir_integral(acum, n_total, volume_assinado, volume_abs):
    """
    Estimador sobre TODAS as tentativas (rejeitadas contam como f = 0), sem reponderação
    pela taxa de aceitação. Combina a estatística dos aceitos com os zeros implícitos:
    soma dos quadrados total = M2_aceitos + |m_aceitos|^2 * k (N - k) / N.
    """
    k = acum.n
    media_total = acum.media * (k / n_total)
    m2_total = acum.m2 + (np.abs(acum.media) ** 2) * (k * (n_total - k) / n_total)
    variancia = m2_total / (n_total - 1)
    integral = volume_assinado * media_total
    erro = volume_abs * np.sqrt(variancia / n_total)
    return {
        "integral": _saida(integral),
        "erro_padrao": _saida(erro),
        "intervalo_confianca_95": (_saida(integral - _Z95 * erro), _saida(integral + _Z95 * erro)),
    }


def _normalizar_difusao(bruto):
    try:
        a = np.asarray(bruto, dtype=float)
    except (TypeError, ValueError):
        raise _ErroSimulacao("a difusão deve ser numérica (escalar, vetor de 3 ou matriz 3×m).") from None
    if a.ndim == 0:
        return float(a) * np.eye(3)
    if a.ndim == 1 and a.shape == (3,):
        return np.diag(a)
    if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] >= 1:
        return a
    raise _ErroSimulacao(
        "a difusão deve ser escalar (σ·I), vetor de 3 (matriz diagonal) ou matriz 3×m.")


class _TrajetoriaInvalida(Exception):
    pass


# ----------------------------------------------------------------------------
# Otimizador inteligente — espaço de busca (Parametro), genes, avaliador e motor
# ----------------------------------------------------------------------------
AMINOACIDOS = "ACDEFGHIKLMNPQRSTVWY"
DNA = "ACGT"
RNA = "ACGU"

_TIPOS_PARAMETRO = ("real", "inteiro", "categorico", "binario", "sequencia",
                    "permutacao", "vetor", "composicao")
_ESCALAS = ("linear", "log")
_CHAMADAS = ("auto", "nomeado", "posicional", "dicionario")
_VIOLACAO_MAXIMA = 1e300
_ROTULOS_MIN = ("minimizar", "min", "minimo", "mínimo")
_ROTULOS_MAX = ("maximizar", "max", "maximo", "máximo")


def _inteiro_exato(valor, descricao):
    if isinstance(valor, bool):
        raise _ErroSimulacao(f"{descricao} deve ser um inteiro.")
    if isinstance(valor, numbers.Integral):
        return int(valor)
    if isinstance(valor, numbers.Real) and math.isfinite(valor) and float(valor).is_integer():
        return int(valor)
    raise _ErroSimulacao(f"{descricao} deve ser um inteiro.")


@dataclass(frozen=True)
class Parametro:
    """
    Descreve UMA variável do espaço de busca do OtimizadorInteligente. Use os construtores:

        Parametro.real(minimo, maximo, escala="linear")   # número real; escala "log" p/ ordens de grandeza
        Parametro.inteiro(minimo, maximo)                 # número inteiro
        Parametro.categorico(escolhas)                    # um item de uma lista (elementos, métodos, ...)
        Parametro.binario()                               # True/False
        Parametro.sequencia(alfabeto, comprimento)        # cadeia: aminoácidos, DNA/RNA, texto, lista
        Parametro.permutacao(itens)                       # ordem dos itens (rotas, ordens de montagem)
        Parametro.vetor(tamanho, minimo, maximo)          # vetor real (pesos, coeficientes, ...)
        Parametro.composicao(tamanho, total=1.0)          # frações >= 0 que somam 'total' (ligas, misturas)

    Atalhos aceitos dentro do dicionário de espaço (sem usar Parametro):
        (0.0, 10.0) -> real (tupla de 2 números é SEMPRE um intervalo real, mesmo com inteiros) |
        range(1, 21) -> inteiro de 1 a 20 (passo 1) | ["Fe", "Cu", "Al"] -> categórico
        (uma LISTA é sempre um conjunto de opções, mesmo com 2 números).

    'comprimento' de sequencia: inteiro (fixo) ou (minimo, maximo) para comprimento variável
    (o otimizador então também insere e remove símbolos). Se o alfabeto for uma string
    (ex.: AMINOACIDOS), a função recebe uma string; senão, recebe uma lista de símbolos.
    O 'tipo' é validado na criação: argumentos inválidos levantam _ErroSimulacao.
    """
    tipo: str
    limites: tuple = ()
    escolhas: tuple = ()
    comprimento: tuple = ()
    tamanho: int = 0
    escala: str = "linear"
    total: float = 1.0
    como_texto: bool = False

    def __post_init__(self):
        def fixar(campo, valor):
            object.__setattr__(self, campo, valor)

        tipo = self.tipo
        if tipo not in _TIPOS_PARAMETRO:
            raise _ErroSimulacao("o tipo do parâmetro deve ser um de: " + ", ".join(_TIPOS_PARAMETRO) + ".")
        if self.escala not in _ESCALAS:
            raise _ErroSimulacao("a escala do parâmetro deve ser 'linear' ou 'log'.")
        if self.escala == "log" and tipo not in ("real", "vetor"):
            raise _ErroSimulacao("a escala 'log' só vale para 'real' e 'vetor'.")

        if tipo in ("real", "vetor"):
            try:
                a, b = (float(x) for x in self.limites)
            except (TypeError, ValueError):
                raise _ErroSimulacao("os limites devem ser um par numérico (minimo, maximo).") from None
            if not (math.isfinite(a) and math.isfinite(b) and a < b):
                raise _ErroSimulacao("os limites devem ser finitos, com minimo < maximo.")
            if self.escala == "log" and a <= 0:
                raise _ErroSimulacao("a escala 'log' exige minimo > 0.")
            fixar("limites", (a, b))
        elif tipo == "inteiro":
            try:
                a, b = self.limites
            except (TypeError, ValueError):
                raise _ErroSimulacao("os limites devem ser um par de inteiros (minimo, maximo).") from None
            a = _inteiro_exato(a, "o mínimo")
            b = _inteiro_exato(b, "o máximo")
            if not a < b:
                raise _ErroSimulacao("os limites devem satisfazer minimo < maximo.")
            fixar("limites", (a, b))

        if tipo in ("vetor", "composicao"):
            n = _inteiro_exato(self.tamanho, "o tamanho")
            minimo_n = 2 if tipo == "composicao" else 1
            if n < minimo_n:
                raise _ErroSimulacao(f"o tamanho deve ser >= {minimo_n}.")
            fixar("tamanho", n)
        if tipo == "composicao":
            total = _validar_real_finito(self.total, "o total da composição")
            if total <= 0:
                raise _ErroSimulacao("o total da composição deve ser > 0.")
            fixar("total", total)

        if tipo == "binario":
            fixar("escolhas", (False, True))
        elif tipo in ("categorico", "sequencia", "permutacao"):
            esc = self.escolhas
            texto = isinstance(esc, str)
            if texto and tipo != "sequencia":
                raise _ErroSimulacao("passe uma lista de opções (uma string só vale como alfabeto de sequência).")
            try:
                esc = tuple(esc)
            except TypeError:
                raise _ErroSimulacao("as opções devem ser uma lista (ou string, para sequências).") from None
            if len(esc) < 2:
                raise _ErroSimulacao("é preciso ao menos 2 opções/símbolos.")
            if tipo in ("sequencia", "permutacao"):
                try:
                    if len(set(esc)) != len(esc):
                        raise _ErroSimulacao("o alfabeto/itens não podem ter repetições.")
                except TypeError:
                    raise _ErroSimulacao("os símbolos/itens precisam ser hashables (números, textos...).") from None
            fixar("escolhas", esc)
            if tipo == "sequencia":
                fixar("como_texto", bool(texto or self.como_texto))

        if tipo == "sequencia":
            c = self.comprimento
            if isinstance(c, (tuple, list)) and len(c) == 2:
                cmin, cmax = _inteiro_exato(c[0], "o comprimento mínimo"), _inteiro_exato(c[1], "o comprimento máximo")
            else:
                cmin = cmax = _inteiro_exato(c, "o comprimento")
            if cmin < 1 or cmax < cmin:
                raise _ErroSimulacao("o comprimento deve ser >= 1 e, se for faixa, (minimo <= maximo).")
            fixar("comprimento", (cmin, cmax))
        if tipo == "permutacao" and len(self.escolhas) < 2:
            raise _ErroSimulacao("uma permutação precisa de ao menos 2 itens.")

    @classmethod
    def real(cls, minimo, maximo, escala="linear"):
        return cls("real", limites=(minimo, maximo), escala=escala)

    @classmethod
    def inteiro(cls, minimo, maximo):
        return cls("inteiro", limites=(minimo, maximo))

    @classmethod
    def categorico(cls, escolhas):
        return cls("categorico", escolhas=escolhas)

    @classmethod
    def binario(cls):
        return cls("binario")

    @classmethod
    def sequencia(cls, alfabeto, comprimento):
        return cls("sequencia", escolhas=alfabeto, comprimento=comprimento)

    @classmethod
    def permutacao(cls, itens):
        return cls("permutacao", escolhas=itens)

    @classmethod
    def vetor(cls, tamanho, minimo, maximo, escala="linear"):
        return cls("vetor", limites=(minimo, maximo), tamanho=tamanho, escala=escala)

    @classmethod
    def composicao(cls, tamanho, total=1.0):
        return cls("composicao", tamanho=tamanho, total=total)


def _parametro_de(spec, nome):
    if isinstance(spec, Parametro):
        return spec
    try:
        if isinstance(spec, range):
            if spec.step != 1 or len(spec) < 2:
                raise _ErroSimulacao("um range de inteiros deve ter passo 1 e ao menos 2 valores.")
            return Parametro.inteiro(spec.start, spec.stop - 1)
        if (isinstance(spec, tuple) and len(spec) == 2
                and all(isinstance(x, numbers.Real) and not isinstance(x, bool) for x in spec)):
            return Parametro.real(*spec)
        if isinstance(spec, (tuple, list)):
            return Parametro.categorico(spec)
    except _ErroSimulacao as erro:
        raise _ErroSimulacao(f"parâmetro '{nome}': {erro}") from None
    raise _ErroSimulacao(
        f"parâmetro '{nome}': especificação inválida ({spec!r}). Use um objeto Parametro, uma tupla "
        "(minimo, maximo) para real, range(a, b + 1) para inteiro ou uma lista de opções.")


def _normalizar_espaco(espaco):
    if isinstance(espaco, dict):
        itens = list(espaco.items())
    else:
        try:
            especs = list(espaco)
        except TypeError:
            raise _ErroSimulacao(
                "'espaco' deve ser um dicionário {nome: Parametro | (min, max) | [opções]} "
                "ou uma lista de pares (min, max).") from None
        itens = [(f"x{i}", s) for i, s in enumerate(especs, start=1)]
    if not itens:
        raise _ErroSimulacao("o espaço de busca não pode ser vazio.")
    nomes, parametros = [], []
    for nome, spec in itens:
        if not isinstance(nome, str) or not nome:
            raise _ErroSimulacao(f"os nomes dos parâmetros devem ser textos não vazios (recebi {nome!r}).")
        parametros.append(_parametro_de(spec, nome))
        nomes.append(nome)
    return nomes, parametros


# ---- genes: como cada tipo de parâmetro é codificado, mutado e recombinado ----------------
def _lim01(u):
    return 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)


class _Gene:
    dim = 1

    def __init__(self, parametro):
        self.p = parametro

    def chave(self, g):
        return g

    def misturar(self, alvo, doador, cr, rng):
        return doador


class _GeneReal(_Gene):
    """g em [0, 1] (normalizado); decodifica para [a, b], linear ou logarítmico."""

    def __init__(self, p):
        super().__init__(p)
        self.a, self.b = p.limites
        self.log = p.escala == "log"
        if self.log:
            self.la, self.lb = math.log(self.a), math.log(self.b)

    def decodificar(self, g):
        v = math.exp(self.la + g * (self.lb - self.la)) if self.log else self.a + g * (self.b - self.a)
        return min(max(v, self.a), self.b)

    def codificar(self, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise _ErroSimulacao("deve ser um número real.") from None
        if not (self.a <= v <= self.b):
            raise _ErroSimulacao(f"{v} está fora dos limites [{self.a}, {self.b}].")
        if self.log:
            return (math.log(v) - self.la) / (self.lb - self.la)
        return (v - self.a) / (self.b - self.a)

    def aleatorio(self, rng):
        return float(rng.random())

    def mutar(self, g, rng, s):
        return _lim01(g + float(rng.normal(0.0, s)))

    def cruzar(self, a, b, rng):
        lo, hi = min(a, b), max(a, b)
        d = hi - lo
        return _lim01(float(rng.uniform(lo - 0.25 * d, hi + 0.25 * d)))

    def diferencial(self, a, b, c, F, rng):
        return _lim01(a + F * (b - c))

    def distancia(self, a, b):
        return abs(a - b)


class _GeneInteiro(_Gene):
    def __init__(self, p):
        super().__init__(p)
        self.lo, self.hi = p.limites

    def decodificar(self, g):
        return int(g)

    def codificar(self, v):
        v = _inteiro_exato(v, "o valor")
        if not (self.lo <= v <= self.hi):
            raise _ErroSimulacao(f"{v} está fora dos limites [{self.lo}, {self.hi}].")
        return v

    def aleatorio(self, rng):
        return int(rng.integers(self.lo, self.hi + 1))

    def _limitar(self, v):
        if v < self.lo:
            v = self.lo + (self.lo - v)
        if v > self.hi:
            v = self.hi - (v - self.hi)
        return int(min(max(v, self.lo), self.hi))

    def mutar(self, g, rng, s):
        d = int(round(float(rng.normal(0.0, max(s * (self.hi - self.lo), 1.0)))))
        if d == 0:
            d = 1 if rng.random() < 0.5 else -1
        return self._limitar(g + d)

    def cruzar(self, a, b, rng):
        return int(rng.integers(min(a, b), max(a, b) + 1))

    def diferencial(self, a, b, c, F, rng):
        d = F * (b - c)
        passo = int(round(d))
        if passo == 0 and b != c:
            passo = 1 if d > 0 else -1
        return self._limitar(a + passo)

    def distancia(self, a, b):
        return abs(a - b) / (self.hi - self.lo)


class _GeneCategorico(_Gene):
    """g é o índice da opção (categórico e binário)."""

    def __init__(self, p):
        super().__init__(p)
        self.k = len(p.escolhas)

    def decodificar(self, g):
        return self.p.escolhas[g]

    def codificar(self, v):
        for i, e in enumerate(self.p.escolhas):
            try:
                igual = bool(e is v or e == v)
            except (TypeError, ValueError):
                igual = False
            if igual:
                return i
        raise _ErroSimulacao(f"{v!r} não está entre as opções {list(self.p.escolhas)!r}.")

    def aleatorio(self, rng):
        return int(rng.integers(self.k))

    def mutar(self, g, rng, s):
        n = int(rng.integers(self.k - 1))
        return n + 1 if n >= g else n

    def cruzar(self, a, b, rng):
        return a if rng.random() < 0.5 else b

    def diferencial(self, a, b, c, F, rng):
        if b != c and rng.random() < min(1.0, F):
            return b
        return a

    def distancia(self, a, b):
        return 0.0 if a == b else 1.0


class _GeneSequencia(_Gene):
    """g é uma tupla de índices do alfabeto; comprimento fixo ou variável."""

    def __init__(self, p):
        super().__init__(p)
        self.k = len(p.escolhas)
        self.lmin, self.lmax = p.comprimento
        self.variavel = self.lmin != self.lmax
        self.dim = self.lmax

    def decodificar(self, g):
        simbolos = [self.p.escolhas[i] for i in g]
        return "".join(simbolos) if self.p.como_texto else simbolos

    def codificar(self, v):
        indice = {s: i for i, s in enumerate(self.p.escolhas)}
        try:
            seq = tuple(indice[s] for s in v)
        except (KeyError, TypeError):
            raise _ErroSimulacao("a sequência contém símbolos fora do alfabeto.") from None
        if not (self.lmin <= len(seq) <= self.lmax):
            raise _ErroSimulacao(f"a sequência tem {len(seq)} símbolos; o comprimento permitido é "
                                 f"[{self.lmin}, {self.lmax}].")
        return seq

    def aleatorio(self, rng):
        n = int(rng.integers(self.lmin, self.lmax + 1))
        return tuple(rng.integers(0, self.k, size=n).tolist())

    def _outro_simbolo(self, atual, rng):
        n = int(rng.integers(self.k - 1))
        return n + 1 if n >= atual else n

    def mutar(self, g, rng, s):
        seq = list(g)
        m = max(1, int(rng.binomial(len(seq), min(1.0, s))))
        for _ in range(m):
            sorteio = rng.random()
            if self.variavel and sorteio < 0.2 and len(seq) < self.lmax:
                seq.insert(int(rng.integers(0, len(seq) + 1)), int(rng.integers(self.k)))
            elif self.variavel and sorteio < 0.4 and len(seq) > self.lmin:
                del seq[int(rng.integers(len(seq)))]
            else:
                pos = int(rng.integers(len(seq)))
                seq[pos] = self._outro_simbolo(seq[pos], rng)
        return tuple(seq)

    def cruzar(self, a, b, rng):
        if len(a) == len(b):
            n = len(a)
            if n == 1:
                return a if rng.random() < 0.5 else b
            i, j = sorted(rng.choice(n + 1, size=2, replace=False).tolist())
            return tuple(a[:i] + b[i:j] + a[j:])
        i = int(rng.integers(0, len(a) + 1))
        j = int(rng.integers(0, len(b) + 1))
        filho = list(a[:i] + b[j:])[:self.lmax]
        while len(filho) < self.lmin:
            filho.append(int(rng.integers(self.k)))
        return tuple(filho)

    def diferencial(self, a, b, c, F, rng):
        novo = list(a)
        for i in range(min(len(a), len(b), len(c))):
            if b[i] != c[i] and rng.random() < F:
                novo[i] = b[i]
        return tuple(novo)

    def distancia(self, a, b):
        m = min(len(a), len(b))
        h = sum(1 for i in range(m) if a[i] != b[i]) + abs(len(a) - len(b))
        return h / max(len(a), len(b))


class _GenePermutacao(_Gene):
    """g é uma permutação (tupla) dos índices dos itens."""

    def __init__(self, p):
        super().__init__(p)
        self.n = len(p.escolhas)
        self.dim = self.n

    def decodificar(self, g):
        return [self.p.escolhas[i] for i in g]

    def codificar(self, v):
        indice = {s: i for i, s in enumerate(self.p.escolhas)}
        try:
            perm = tuple(indice[s] for s in v)
        except (KeyError, TypeError):
            raise _ErroSimulacao("a ordem contém itens que não pertencem à lista.") from None
        if sorted(perm) != list(range(self.n)):
            raise _ErroSimulacao("a ordem deve conter cada item exatamente uma vez.")
        return perm

    def aleatorio(self, rng):
        return tuple(rng.permutation(self.n).tolist())

    def mutar(self, g, rng, s):
        perm = list(g)
        for _ in range(1 + (1 if rng.random() < min(1.0, s) else 0)):
            tipo = int(rng.integers(3))
            i, j = sorted(rng.choice(self.n, size=2, replace=False).tolist())
            if tipo == 0:
                perm[i], perm[j] = perm[j], perm[i]
            elif tipo == 1:
                perm[i:j + 1] = perm[i:j + 1][::-1]
            else:
                perm.insert(j, perm.pop(i))
        return tuple(perm)

    def cruzar(self, a, b, rng):
        i, j = sorted(rng.choice(self.n + 1, size=2, replace=False).tolist())
        meio = a[i:j]
        usados = set(meio)
        resto = [x for x in b if x not in usados]
        return tuple(resto[:i] + list(meio) + resto[i:])

    def diferencial(self, a, b, c, F, rng):
        if b != c and rng.random() < min(1.0, F):
            return self.cruzar(a, b, rng)
        return a

    def distancia(self, a, b):
        return sum(1 for x, y in zip(a, b) if x != y) / self.n


class _GeneVetor(_Gene):
    """g é uma tupla de n valores em [0, 1] (normalizados); decodifica para um ndarray."""

    def __init__(self, p):
        super().__init__(p)
        self.n = p.tamanho
        self.dim = self.n
        if p.tipo == "vetor":
            self.a, self.b = p.limites
            self.log = p.escala == "log"
            if self.log:
                self.la, self.lb = math.log(self.a), math.log(self.b)

    def decodificar(self, g):
        u = np.asarray(g, dtype=float)
        v = np.exp(self.la + u * (self.lb - self.la)) if self.log else self.a + u * (self.b - self.a)
        return np.clip(v, self.a, self.b)

    def codificar(self, v):
        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            raise _ErroSimulacao("deve ser um vetor numérico.") from None
        if arr.shape != (self.n,):
            raise _ErroSimulacao(f"deve ter exatamente {self.n} componentes.")
        if not np.all((arr >= self.a) & (arr <= self.b)):
            raise _ErroSimulacao(f"todas as componentes devem estar em [{self.a}, {self.b}].")
        u = (np.log(arr) - self.la) / (self.lb - self.la) if self.log else (arr - self.a) / (self.b - self.a)
        return tuple(np.clip(u, 0.0, 1.0).tolist())

    def aleatorio(self, rng):
        return tuple(rng.random(self.n).tolist())

    def mutar(self, g, rng, s):
        return tuple(np.clip(np.asarray(g) + rng.normal(0.0, s, self.n), 0.0, 1.0).tolist())

    def cruzar(self, a, b, rng):
        a, b = np.asarray(a), np.asarray(b)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        d = hi - lo
        return tuple(np.clip(rng.uniform(lo - 0.25 * d, hi + 0.25 * d), 0.0, 1.0).tolist())

    def diferencial(self, a, b, c, F, rng):
        return tuple(np.clip(np.asarray(a) + F * (np.asarray(b) - np.asarray(c)), 0.0, 1.0).tolist())

    def misturar(self, alvo, doador, cr, rng):
        mascara = rng.random(self.n) < cr
        mascara[int(rng.integers(self.n))] = True
        return tuple(np.where(mascara, np.asarray(doador), np.asarray(alvo)).tolist())

    def distancia(self, a, b):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


class _GeneComposicao(_GeneVetor):
    """g são pesos >= 0 em [0, 1]; a composição é g normalizado para somar 'total'."""

    def __init__(self, p):
        _Gene.__init__(self, p)
        self.n = p.tamanho
        self.dim = self.n
        self.total = p.total

    def _fracoes(self, g):
        u = np.maximum(np.asarray(g, dtype=float), 0.0)
        s = float(u.sum())
        return np.full(self.n, 1.0 / self.n) if s <= 0 else u / s

    def decodificar(self, g):
        return self._fracoes(g) * self.total

    def codificar(self, v):
        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            raise _ErroSimulacao("deve ser um vetor numérico.") from None
        if arr.shape != (self.n,) or not np.all(np.isfinite(arr)) or np.any(arr < 0) or arr.sum() <= 0:
            raise _ErroSimulacao(f"deve ter {self.n} componentes finitas, >= 0 e soma positiva.")
        p = arr / arr.sum()
        return tuple((p / p.max()).tolist())

    def aleatorio(self, rng):
        e = rng.exponential(1.0, self.n)
        if rng.random() < 0.3:
            e[rng.choice(self.n, size=int(rng.integers(1, self.n)), replace=False)] = 0.0
        return tuple((e / e.max()).tolist())

    def distancia(self, a, b):
        return float(0.5 * np.sum(np.abs(self._fracoes(a) - self._fracoes(b))))


def _criar_gene(p):
    classe = {"real": _GeneReal, "inteiro": _GeneInteiro, "categorico": _GeneCategorico,
              "binario": _GeneCategorico, "sequencia": _GeneSequencia, "permutacao": _GenePermutacao,
              "vetor": _GeneVetor, "composicao": _GeneComposicao}[p.tipo]
    return classe(p)


def _codificar_iniciais(valores, nomes, genes, rng):
    if valores is None:
        return []
    if isinstance(valores, dict):
        valores = [valores]
    try:
        lista = list(valores)
    except TypeError:
        raise _ErroSimulacao("'valores_iniciais' deve ser um dicionário {nome: valor} ou uma lista deles.") from None
    saida = []
    for d in lista:
        if not isinstance(d, dict):
            raise _ErroSimulacao("cada valor inicial deve ser um dicionário {nome: valor}.")
        desconhecidos = [k for k in d if k not in nomes]
        if desconhecidos:
            raise _ErroSimulacao(f"valores iniciais para parâmetros que não existem: {desconhecidos}.")
        genoma = []
        for nome, gene in zip(nomes, genes):
            if nome in d:
                try:
                    genoma.append(gene.codificar(d[nome]))
                except _ErroSimulacao as erro:
                    raise _ErroSimulacao(f"valor inicial de '{nome}': {erro}") from None
            else:
                genoma.append(gene.aleatorio(rng))
        saida.append(genoma)
    return saida


# ---- como as funções do usuário recebem os parâmetros -------------------------------------
def _inferir_chamada(f, nomes, fixos, rotulo):
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        raise _ErroSimulacao(
            f"não consegui inspecionar a assinatura de {rotulo}; informe chamada='nomeado', "
            "'posicional' ou 'dicionario'.") from None
    ps = list(sig.parameters.values())
    tem_kwargs = any(p.kind is p.VAR_KEYWORD for p in ps)
    tem_varargs = any(p.kind is p.VAR_POSITIONAL for p in ps)
    nomeados = [p for p in ps if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    obrigatorios = [p for p in nomeados if p.default is p.empty]
    pos_obrigatorios = [p for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                        and p.default is p.empty]
    disponiveis = set(nomes) | set(fixos)
    if tem_kwargs:
        return "nomeado"
    if nomeados and any(p.name in disponiveis for p in nomeados) \
            and all(p.name in disponiveis for p in obrigatorios):
        return "nomeado"
    if tem_varargs and not nomeados:
        return "posicional"
    if len(pos_obrigatorios) == len(nomes):
        return "posicional"
    if len(pos_obrigatorios) == 1:
        return "dicionario"
    raise _ErroSimulacao(
        f"não consegui decidir como passar os parâmetros {nomes} para {rotulo}. Use argumentos com "
        "os mesmos nomes dos parâmetros, ou informe chamada='nomeado', 'posicional' ou 'dicionario'.")


def _construir_chamador(f, chamada, nomes, fixos, rotulo):
    """Devolve (chamar(params_dict), modo_usado, nomes_efetivamente_enviados)."""
    modo = _inferir_chamada(f, nomes, fixos, rotulo) if chamada == "auto" else chamada
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        sig = None

    if modo == "nomeado":
        aceita = None
        if sig is not None:
            ps = list(sig.parameters.values())
            if not any(p.kind is p.VAR_KEYWORD for p in ps):
                aceita = {p.name for p in ps if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        enviados = [n for n in list(fixos) + list(nomes) if aceita is None or n in aceita]
        if sig is not None:
            try:
                sig.bind(**{n: None for n in enviados})
            except TypeError as erro:
                raise _ErroSimulacao(
                    f"{rotulo} não aceita os argumentos nomeados {enviados} ({erro}).") from None

        def chamar(params):
            kw = dict(fixos)
            kw.update(params)
            if aceita is not None:
                kw = {k: v for k, v in kw.items() if k in aceita}
            return f(**kw)
        return chamar, modo, {n for n in nomes if n in enviados}

    if modo == "dicionario":
        if sig is not None:
            try:
                sig.bind(None)
            except TypeError as erro:
                raise _ErroSimulacao(f"{rotulo} deve aceitar UM argumento (o dicionário de parâmetros): {erro}.") from None

        def chamar(params):
            d = dict(fixos)
            d.update(params)
            return f(d)
        return chamar, modo, set(nomes)

    if fixos:
        raise _ErroSimulacao("'fixos' só pode ser usado com chamada 'nomeado' ou 'dicionario'.")
    if sig is not None:
        try:
            sig.bind(*([None] * len(nomes)))
        except TypeError as erro:
            raise _ErroSimulacao(
                f"{rotulo} deve aceitar {len(nomes)} argumentos posicionais (um por parâmetro, na ordem "
                f"{nomes}): {erro}.") from None

    def chamar(params):
        return f(*[params[n] for n in nomes])
    return chamar, modo, set(nomes)


# ---- alvo, pesos ------------------------------------------------------------------------
def _termo_do_alvo(spec):
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in _ROTULOS_MIN:
            return ("minimizar", None)
        if s in _ROTULOS_MAX:
            return ("maximizar", None)
        raise _ErroSimulacao("o alvo textual deve ser 'minimizar' ou 'maximizar' "
                             "(ou um número/vetor, para atingir um valor).")
    try:
        a = np.asarray(spec, dtype=float)
    except (TypeError, ValueError):
        raise _ErroSimulacao(f"alvo inválido ({spec!r}): use 'minimizar', 'maximizar' ou um número/vetor real.") from None
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise _ErroSimulacao(f"alvo inválido ({spec!r}): o valor-alvo deve ser finito e não vazio.")
    return ("valor", a)


def _normalizar_alvo(alvo, n):
    if n == 1:
        return [_termo_do_alvo(alvo)]
    if isinstance(alvo, (list, tuple)) or (isinstance(alvo, np.ndarray) and alvo.ndim >= 1):
        if len(alvo) == n:
            return [_termo_do_alvo(a) for a in alvo]
    elif isinstance(alvo, str) or (isinstance(alvo, numbers.Real) and not isinstance(alvo, bool)):
        return [_termo_do_alvo(alvo) for _ in range(n)]
    raise _ErroSimulacao(
        f"com {n} grandezas, 'alvo' deve ser um texto/número único (vale para todas) ou uma lista com "
        f"exatamente {n} entradas (uma por grandeza: 'minimizar', 'maximizar' ou número/vetor).")


def _normalizar_pesos(pesos, n):
    if pesos is None:
        return [1.0] * n
    if isinstance(pesos, numbers.Real) and not isinstance(pesos, bool):
        pesos = [pesos]
    try:
        lista = [float(p) for p in pesos]
    except (TypeError, ValueError):
        raise _ErroSimulacao("'pesos' deve ser uma lista de números positivos (um por grandeza).") from None
    if len(lista) != n or not all(math.isfinite(p) and p > 0 for p in lista):
        raise _ErroSimulacao(f"'pesos' deve ter exatamente {n} números finitos e positivos.")
    return lista


def _como_array_real(valor):
    a = _como_array(valor)
    if a.dtype.kind == "c":
        raise _ErroSimulacao("o otimizador exige saídas reais; para saídas complexas passe uma função "
                             "por componente (parte real e imaginária).")
    return a


# ---- avaliador: parâmetros -> (violação, perda, alvo_ok) -----------------------------------
class _Avaliador:
    """
    Registro de uma avaliação: (violacao, perda, alvo_ok).
      violacao = 0 (viável) | > 0 (soma das violações das condições) | inf (avaliação indefinida)
      perda    = escalarização a MINIMIZAR (inf se a função não foi avaliada)
      alvo_ok  = True/False se TODAS as grandezas têm alvo de valor; None caso contrário
    """

    def __init__(self, genes, nomes, chamadores, combinar, termos, pesos, condicoes, invalidos,
                 repeticoes, usar_cache, tolerancia_alvo):
        self.genes = genes
        self.nomes = nomes
        self.chamadores = chamadores
        self.combinar = combinar
        self.termos = termos
        self.pesos = pesos
        self.condicoes = condicoes
        self.invalidos = invalidos
        self.repeticoes = repeticoes
        self.usar_cache = usar_cache
        self.tol = tolerancia_alvo
        self.cache = {}
        self.sem_gradiente = False
        self.todos_valor = all(t[0] == "valor" for t in termos)

    def chave(self, genoma):
        return tuple(g.chave(x) for g, x in zip(self.genes, genoma))

    def parametros(self, genoma):
        return {n: g.decodificar(x) for n, g, x in zip(self.nomes, self.genes, genoma)}

    def valores(self, params):
        """(saidas, combinado) como arrays reais finitos, ou None se indefinido/não finito."""
        saidas = []
        try:
            with np.errstate(all="ignore"):
                for chamar in self.chamadores:
                    acc = None
                    for _ in range(self.repeticoes):
                        a = _como_array_real(chamar(params)).astype(float)
                        acc = a if acc is None else acc + a
                    saidas.append(acc / self.repeticoes)
                combinado = None
                if self.combinar is not None:
                    combinado = _como_array_real(self.combinar(*saidas)).astype(float)
        except _EXCECOES_NUMERICAS:
            return None
        for a in saidas + ([combinado] if combinado is not None else []):
            if not np.all(np.isfinite(a)):
                return None
        return saidas, combinado

    def contribuicoes(self, saidas, combinado):
        quantidades = [combinado] if self.combinar is not None else saidas
        itens = []
        for q, (tipo, alvo), w in zip(quantidades, self.termos, self.pesos):
            if tipo == "valor":
                if alvo.ndim > 0 and q.shape != alvo.shape:
                    raise _ErroSimulacao(
                        f"o resultado tem formato {q.shape}, mas o alvo tem formato {alvo.shape}; devem coincidir.")
                dev = q - alvo
                eqm = float(np.mean(dev ** 2))
                itens.append((w * eqm, float(np.max(np.abs(dev))), eqm))
            else:
                if q.size != 1:
                    raise _ErroSimulacao(
                        "para minimizar/maximizar, a função deve retornar um escalar real; para ajustar uma "
                        "saída vetorial a dados, passe alvo=<vetor com os dados>.")
                v = float(q.reshape(()))
                itens.append((w * (v if tipo == "minimizar" else -v), None, None))
        return itens

    def _calcular(self, params):
        viol = 0.0
        for c in self.condicoes:
            margem, numerica = c.avaliar(params)
            if margem < 0:
                viol += _VIOLACAO_MAXIMA if margem == -math.inf else -margem
                if not numerica:
                    self.sem_gradiente = True
        if viol > 0:
            return (min(viol, _VIOLACAO_MAXIMA), math.inf, None), None
        res = self.valores(params)
        perda = math.inf
        itens = None
        if res is not None:
            itens = self.contribuicoes(*res)
            perda = float(sum(i[0] for i in itens))
        if res is None or not math.isfinite(perda):
            if self.invalidos == "erro":
                raise _ErroSimulacao(
                    f"a função é indefinida ou não finita nos parâmetros {params}. Use "
                    "invalidos='ignorar' para descartar esses candidatos.")
            return (math.inf, math.inf, None), None
        ok = None
        if self.todos_valor:
            ok = all(i[1] <= self.tol for i in itens)
        return (0.0, perda, ok), res

    def __call__(self, genoma):
        chave = self.chave(genoma)
        if self.usar_cache and chave in self.cache:
            return self.cache[chave], None, False
        reg, payload = self._calcular(self.parametros(genoma))
        if self.usar_cache:
            self.cache[chave] = reg
        return reg, payload, True


# ---- motor evolutivo adaptativo ---------------------------------------------------------
def _melhor_que(a, b):
    """a é ESTRITAMENTE melhor que b? Regras de Deb: menor violação vence; em empate, menor perda."""
    if a[0] != b[0]:
        return a[0] < b[0]
    return a[1] < b[1]


class _MotorEvolutivo:
    """
    Algoritmo evolutivo híbrido com mistura de operadores.

    População de candidatos; a cada tentativa um operador é sorteado (DE/rand/1, DE/atual-para-melhor,
    cruzamento+mutação, mutação com passo auto-adaptativo, busca local no melhor). A probabilidade de
    cada operador tem um piso (_PMIN) e o restante é redistribuído por probability matching sobre a
    recompensa recente (melhorar o candidato-base / o melhor global). Nos testes, esse aprendizado
    adicionou pouco: o sorteio uniforme foi comparável ou melhor em problemas discretos. O passo
    das mutações (sigma) é auto-adaptado de fato (regra 1/5 de sucessos). O candidato gerado substitui sua base se for >= (regras de Deb
    para restrições: viável sempre vence inviável; entre inviáveis, menor violação), então o melhor
    nunca é perdido.

    Reinícios independentes (estilo IPOP): quando uma fase estagna, o melhor global é guardado à
    parte e uma nova fase recomeça do zero com população maior (x1,5, até 6x). A busca termina antes
    do orçamento apenas se o MESMO ótimo foi reencontrado por 'confirmacoes' fases independentes
    (evidência de convergência, não prova) ou se os reinícios acabaram.
    """
    OPERADORES = ("de_rand_1", "de_atual_para_melhor", "cruzamento", "mutacao_adaptativa", "busca_local")
    # Piso de probabilidade por operador. Com 5 operadores, 0.20 seria sorteio uniforme (sem aprendizado).
    # Medido: com 0.05 o 'aprendizado' favorecia operadores que melhoram indivíduos ruins e piorava
    # problemas discretos; 0.16 foi o melhor compromisso nos testes (80% do peso uniforme, 20% adaptativo).
    _PMIN = 0.16
    _ALFA = 0.15
    _SIGMAS_INICIAIS = {"mut": 0.15, "local": 0.03}
    _LIMITES_SIGMA = {"mut": (1e-7, 0.5), "local": (1e-9, 0.1)}

    def __init__(self, genes, avaliador, rng, max_avaliacoes, tamanho_populacao, paciencia,
                 max_reinicios, confirmacoes, iniciais, num_melhores, guardar_historico):
        self.genes = genes
        self.ng = len(genes)
        self.av = avaliador
        self.rng = rng
        self.max_av = max_avaliacoes
        self.N = tamanho_populacao
        self.paciencia = paciencia
        self.max_reinicios = max_reinicios
        self.confirmacoes = confirmacoes
        self.N_max = min(6 * tamanho_populacao, max(tamanho_populacao, max_avaliacoes // 4))
        self.fases = []
        self.confirmadas = 0
        self.iniciais = iniciais
        self.num_melhores = num_melhores
        self.historico = [] if guardar_historico else None
        self.max_tentativas = 30 * max_avaliacoes + 1000
        K = len(self.OPERADORES)
        self.q = np.ones(K)
        self.usos = [0] * K
        self.sucessos = [0] * K
        self.novos_melhores = [0] * K
        self.sigma = dict(self._SIGMAS_INICIAIS)
        self.pop, self.reg = [], []
        self.melhor = None
        self.elite = {}
        self.n_aval = self.n_cache = self.n_indefinidas = self.n_inviaveis = self.tentativas = 0
        self.reinicios = self.geracoes = self.sem_melhora_final = 0
        self.diversidade_final = 0.0
        self.motivo = None
        self._melhoria = False
        self._ops = [self._op_de_rand_1, self._op_de_atual_para_melhor, self._op_cruzamento,
                     self._op_mutacao, self._op_busca_local]

    # -------------------------------------------------------------- utilidades
    def _chave(self, genoma):
        return self.av.chave(genoma)

    def _aleatorio(self):
        return [g.aleatorio(self.rng) for g in self.genes]

    def _indice_melhor(self):
        return min(range(len(self.pop)), key=lambda j: (self.reg[j][0], self.reg[j][1]))

    def _tres_distintos(self, i, quantos):
        n = len(self.pop)
        while True:
            r = self.rng.integers(0, n, size=quantos).tolist()
            if len(set(r) | {i}) == quantos + 1:
                return r

    def _torneio(self, tamanho=3):
        n = len(self.pop)
        candidatos = self.rng.integers(0, n, size=tamanho).tolist()
        return min(candidatos, key=lambda j: (self.reg[j][0], self.reg[j][1]))

    def _mutar(self, base, sigma, p_gene):
        mascara = self.rng.random(self.ng) < p_gene
        if not mascara.any():
            mascara[int(self.rng.integers(self.ng))] = True
        return [g.mutar(base[j], self.rng, sigma) if mascara[j] else base[j]
                for j, g in enumerate(self.genes)]

    def _diversidade(self):
        n = len(self.pop)
        if n < 2:
            return 0.0
        pares = min(20, n * (n - 1) // 2)
        total = 0.0
        for _ in range(pares):
            a, b = self.rng.choice(n, size=2, replace=False).tolist()
            total += sum(g.distancia(self.pop[a][j], self.pop[b][j])
                         for j, g in enumerate(self.genes)) / self.ng
        return total / pares

    @staticmethod
    def _melhora_significativa(novo, antigo):
        if antigo[0] == math.inf:
            return novo[0] != math.inf
        if novo[0] != antigo[0]:
            return antigo[0] - novo[0] > max(1e-12, 1e-9 * antigo[0])
        if math.isinf(antigo[1]):
            return not math.isinf(novo[1])
        return antigo[1] - novo[1] > max(1e-12, 1e-9 * abs(antigo[1]))

    # -------------------------------------------------------------- avaliação e registro
    def _considerar(self, genoma, reg, payload):
        if self.melhor is None or _melhor_que(reg, self.melhor[1]):
            if self.melhor is None or self._melhora_significativa(reg, self.melhor[1]):
                self._melhoria = True
            self.melhor = (genoma, reg, payload)
        if reg[0] == 0.0:
            self.elite[self._chave(genoma)] = (reg, genoma, payload)
            if len(self.elite) > self.num_melhores:
                pior = max(self.elite, key=lambda k: (self.elite[k][0][0], self.elite[k][0][1]))
                del self.elite[pior]

    def _avaliar(self, genoma):
        self.tentativas += 1
        reg, payload, nova = self.av(genoma)
        if nova:
            self.n_aval += 1
            if reg[0] == math.inf:
                self.n_indefinidas += 1
            elif reg[0] > 0:
                self.n_inviaveis += 1
            self._considerar(genoma, reg, payload)
        else:
            self.n_cache += 1
        return reg

    # -------------------------------------------------------------- operadores
    def _op_de_rand_1(self, i):
        r1, r2, r3 = self._tres_distintos(i, 3)
        F = float(self.rng.uniform(0.4, 0.95))
        CR = float(self.rng.uniform(0.1, 1.0))
        alvo = self.pop[i]
        j0 = int(self.rng.integers(self.ng))
        trial = []
        for j, g in enumerate(self.genes):
            if j == j0 or self.rng.random() < CR:
                d = g.diferencial(self.pop[r1][j], self.pop[r2][j], self.pop[r3][j], F, self.rng)
                trial.append(g.misturar(alvo[j], d, CR, self.rng))
            else:
                trial.append(alvo[j])
        return trial, i

    def _op_de_atual_para_melhor(self, i):
        r1, r2 = self._tres_distintos(i, 2)
        melhor = self.pop[self._indice_melhor()]
        F = float(self.rng.uniform(0.4, 0.95))
        CR = float(self.rng.uniform(0.1, 1.0))
        alvo = self.pop[i]
        j0 = int(self.rng.integers(self.ng))
        trial = []
        for j, g in enumerate(self.genes):
            if j == j0 or self.rng.random() < CR:
                a = g.diferencial(alvo[j], melhor[j], alvo[j], F, self.rng)
                d = g.diferencial(a, self.pop[r1][j], self.pop[r2][j], F, self.rng)
                trial.append(g.misturar(alvo[j], d, CR, self.rng))
            else:
                trial.append(alvo[j])
        return trial, i

    def _op_cruzamento(self, i):
        p1 = self._torneio()
        p2 = self._torneio()
        for _ in range(3):
            if p2 != p1:
                break
            p2 = self._torneio()
        filho = [g.cruzar(self.pop[p1][j], self.pop[p2][j], self.rng) for j, g in enumerate(self.genes)]
        for j, g in enumerate(self.genes):
            if self.rng.random() < 1.0 / self.ng:
                filho[j] = g.mutar(filho[j], self.rng, self.sigma["mut"])
        chave = self._chave(filho)
        if chave in (self._chave(self.pop[p1]), self._chave(self.pop[p2])):
            j = int(self.rng.integers(self.ng))
            filho[j] = self.genes[j].mutar(filho[j], self.rng, self.sigma["mut"])
        pior = p2 if _melhor_que(self.reg[p1], self.reg[p2]) else p1
        return filho, pior

    def _op_mutacao(self, i):
        return self._mutar(self.pop[i], self.sigma["mut"], max(1.0 / self.ng, 0.2)), i

    def _op_busca_local(self, i):
        ib = self._indice_melhor()
        return self._mutar(self.pop[ib], self.sigma["local"], max(1.0 / self.ng, 0.1)), ib

    # -------------------------------------------------------------- aprendizado de operadores
    def _escolher_operador(self):
        K = len(self.OPERADORES)
        s = float(self.q.sum())
        p = np.full(K, 1.0 / K) if s <= 0 else self._PMIN + (1.0 - K * self._PMIN) * self.q / s
        return int(self.rng.choice(K, p=p / p.sum()))

    def probabilidades(self):
        K = len(self.OPERADORES)
        s = float(self.q.sum())
        return (np.full(K, 1.0 / K) if s <= 0 else self._PMIN + (1.0 - K * self._PMIN) * self.q / s).tolist()

    def _passo(self, i):
        k = self._escolher_operador()
        trial, base = self._ops[k](i)
        reg_base = self.reg[base]
        melhor_antes = self.melhor[1]
        reg = self._avaliar(trial)
        melhor_que_base = _melhor_que(reg, reg_base)
        if not _melhor_que(reg_base, reg) and self._chave(trial) != self._chave(self.pop[base]):
            self.pop[base] = trial
            self.reg[base] = reg
        recompensa = 0.0
        if melhor_que_base:
            recompensa = 1.0
        novo_melhor = _melhor_que(reg, melhor_antes)
        if novo_melhor:
            recompensa = 2.0
        self.usos[k] += 1
        self.sucessos[k] += int(melhor_que_base)
        self.novos_melhores[k] += int(novo_melhor)
        self.q[k] = (1.0 - self._ALFA) * self.q[k] + self._ALFA * recompensa
        if k in (3, 4):
            nome = "mut" if k == 3 else "local"
            fator = 1.2 if melhor_que_base else 1.2 ** -0.25
            lo, hi = self._LIMITES_SIGMA[nome]
            self.sigma[nome] = min(hi, max(lo, self.sigma[nome] * fator))

    # -------------------------------------------------------------- ciclo de vida
    def _inicializar(self):
        genomas = list(self.iniciais)
        while len(genomas) < self.N:
            genomas.append(self._aleatorio())
        for g in genomas:
            if self.n_aval >= self.max_av:
                break
            reg = self._avaliar(g)
            self.pop.append(g)
            self.reg.append(reg)

    def _melhor_da_populacao(self):
        return self.reg[self._indice_melhor()]

    def _fechar_fase(self, completa=True):
        reg = self._melhor_da_populacao()
        self.fases.append({"tamanho_populacao": len(self.pop), "violacao": reg[0],
                           "perda": reg[1] if math.isfinite(reg[1]) else None,
                           "avaliacoes_acumuladas": self.n_aval, "completa": completa})

    def _contar_confirmacoes(self):
        """Quantas fases terminaram no mesmo valor do melhor global (tolerância relativa 1e-6)."""
        g = self.melhor[1]
        if g[0] != 0.0 or not math.isfinite(g[1]):
            return 0
        n = 0
        for fase in self.fases:
            if fase["completa"] and fase["violacao"] == 0.0 and fase["perda"] is not None:
                if abs(fase["perda"] - g[1]) <= 1e-6 * max(1.0, abs(fase["perda"]), abs(g[1])):
                    n += 1
        return n

    def _reiniciar(self):
        """Nova fase independente: população aleatória maior; o melhor global fica guardado à parte."""
        novo_n = min(int(math.ceil(len(self.pop) * 1.5)), self.N_max)
        pop, regs = [], []
        for _ in range(novo_n):
            if self.n_aval >= self.max_av:
                break
            g = self._aleatorio()
            pop.append(g)
            regs.append(self._avaliar(g))
        if pop:
            self.pop, self.reg = pop, regs
        self.sigma = dict(self._SIGMAS_INICIAIS)

    def _alvo_atingido(self):
        return self.melhor[1][2] is True

    def _registrar_historico(self, diversidade):
        if self.historico is not None:
            self.historico.append({"avaliacoes": self.n_aval, "fase": self.reinicios + 1,
                                   "violacao": self.melhor[1][0], "perda": self.melhor[1][1],
                                   "diversidade": diversidade})

    def executar(self):
        self._inicializar()
        sem_melhora_fase = sem_melhora_global = 0
        ref_fase = self._melhor_da_populacao()
        diversidade = self._diversidade()
        self._registrar_historico(diversidade)
        while self.motivo is None:
            if self._alvo_atingido():
                self.motivo = "alvo atingido dentro da tolerância"
                break
            self.geracoes += 1
            self._melhoria = False
            for i in self.rng.permutation(len(self.pop)).tolist():
                if self.n_aval >= self.max_av or self.tentativas >= self.max_tentativas:
                    break
                self._passo(i)
                if self._alvo_atingido():
                    break
            sem_melhora_global = 0 if self._melhoria else sem_melhora_global + 1
            atual = self._melhor_da_populacao()
            if self._melhora_significativa(atual, ref_fase):
                sem_melhora_fase = 0
                ref_fase = atual
            else:
                sem_melhora_fase += 1
            diversidade = self._diversidade()
            self._registrar_historico(diversidade)
            if self._alvo_atingido():
                self.motivo = "alvo atingido dentro da tolerância"
            elif self.n_aval >= self.max_av:
                self.motivo = "orçamento de avaliações esgotado"
            elif self.tentativas >= self.max_tentativas:
                self.motivo = ("limite de tentativas: o espaço parece pequeno demais, a maioria dos "
                               "candidatos gerados já havia sido avaliada")
            elif sem_melhora_fase >= self.paciencia or (diversidade < 1e-4 and sem_melhora_fase >= 5):
                self._fechar_fase()
                self.confirmadas = self._contar_confirmacoes()
                if self.confirmadas >= self.confirmacoes:
                    self.motivo = (f"o mesmo ótimo foi reencontrado por {self.confirmadas} fases "
                                   "independentes")
                elif self.reinicios >= self.max_reinicios:
                    self.motivo = ("reinícios esgotados" if self.max_reinicios > 0 else
                                   f"sem melhora significativa em {self.paciencia} gerações "
                                   "(reinícios desativados)")
                else:
                    self._reiniciar()
                    self.reinicios += 1
                    sem_melhora_fase = 0
                    ref_fase = self._melhor_da_populacao()
        if not self.fases or self.fases[-1]["avaliacoes_acumuladas"] != self.n_aval:
            self._fechar_fase(completa=False)
        self.confirmadas = self._contar_confirmacoes()
        self.sem_melhora_final = sem_melhora_global
        self.diversidade_final = diversidade


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
class SimulacoesSolver:
    """
    Módulo de Simulações e Otimização Genéricas do OMEGA.
    Projetado para aceitar qualquer tipo de entrada avaliável numericamente
    (funções, equações combinadas, integrais, derivadas, etc.), em número arbitrário,
    com condições de contorno arbitrárias (veja o docstring do módulo e a classe Condicao).
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ integração
    def _integrar_n_dimensoes(self, funcoes, limites, num_amostras, condicoes,
                              combinar, invalidos, semente):
        pares = _validar_limites(limites)
        num_amostras = _validar_inteiro(num_amostras, "o número de amostras", 2)
        _validar_politica(invalidos)
        _validar_combinar(combinar)
        lista_cond = _normalizar_condicoes(condicoes, permitir_igualdade=False)

        inferior = np.array([min(a, b) for a, b in pares])
        superior = np.array([max(a, b) for a, b in pares])
        volume_abs = float(np.prod(superior - inferior))
        sinal = math.prod(1.0 if b >= a else -1.0 for a, b in pares)
        rng = np.random.default_rng(semente)
        avisos = []

        n_quantidades = len(funcoes) + (1 if combinar is not None else 0)
        acumuladores = [_Acumulador() for _ in range(n_quantidades)]
        aceitas = invalidas = 0

        for ponto in _blocos_de_pontos(rng, inferior, superior, num_amostras):
            if lista_cond and _margem_minima(lista_cond, ponto, parar_se_violada=True) < 0:
                continue
            saidas = _avaliar_funcoes(funcoes, combinar, ponto)
            if saidas is None:
                if invalidos == "erro":
                    raise _ErroSimulacao(
                        f"alguma função é indefinida ou não finita no ponto {ponto}. Use "
                        "invalidos='ignorar' para descartá-lo (isso pode enviesar o resultado).")
                invalidas += 1
                continue
            for acum, s in zip(acumuladores, saidas):
                acum.adicionar(s)
            aceitas += 1

        n_validas = num_amostras - invalidas
        if n_validas < 2:
            raise _ErroSimulacao("restaram menos de 2 amostras válidas; não há como estimar a integral.")
        if invalidas:
            _avisar(avisos, f"{invalidas} de {num_amostras} amostras foram descartadas por avaliação "
                            "indefinida/não finita; se não forem desprezíveis, a estimativa é enviesada.")
        p = aceitas / n_validas
        volume = {"estimado": volume_abs * p,
                  "erro_padrao": volume_abs * math.sqrt(p * (1 - p) / (n_validas - 1))}

        if aceitas == 0:
            _avisar(avisos, "nenhuma amostra caiu no domínio admissível; a integral estimada é 0 sem erro "
                            "padrão calculável. Aumente num_amostras ou revise as condições.")
            quantidades = [{"integral": 0.0, "erro_padrao": None, "intervalo_confianca_95": None}
                           for _ in acumuladores]
        else:
            if aceitas < 30:
                _avisar(avisos, f"apenas {aceitas} amostras aceitas: o erro padrão e o intervalo de "
                                "confiança (baseados no TCL) não são confiáveis.")
            quantidades = [_resumir_integral(a, n_validas, sinal * volume_abs, volume_abs)
                           for a in acumuladores]

        por_funcao = quantidades[:len(funcoes)]
        combinado = quantidades[len(funcoes)] if combinar is not None else None
        if combinado is not None:
            resultado = dict(combinado)
        elif len(funcoes) == 1:
            resultado = dict(por_funcao[0])
        else:
            resultado = {
                "integral": [q["integral"] for q in por_funcao],
                "erro_padrao": [q["erro_padrao"] for q in por_funcao],
                "intervalo_confianca_95": [q["intervalo_confianca_95"] for q in por_funcao],
            }
        resultado.update({
            "por_funcao": por_funcao,
            "combinado": combinado,
            "volume_dominio": volume,
            "fracao_aceita": p,
            "amostras": {"tentativas": num_amostras, "validas": n_validas,
                         "aceitas": aceitas, "invalidas": invalidas},
            "avisos": avisos,
            "limitacoes": [
                "Estimativa estocástica: o erro padrão decai como N^(-1/2); o intervalo de 95% "
                "assume normalidade assintótica (TCL) e exige variância finita e muitas amostras aceitas.",
                "Amostragem uniforme na caixa de 'limites': singularidades integráveis mas fortes, ou "
                "domínios com fração aceita minúscula, geram variância alta ou infinita.",
            ],
        })
        return resultado

    @_retorna_mensagem_de_erro
    def MonteCarloIntegracao1D(self, f, a, b, *funcoes_adicionais, num_amostras=100000,
                               condicoes=(), combinar=None, semente=None, invalidos="ignorar"):
        """
        Integral definida em [a, b] por Monte Carlo (média amostral), para n funções ao mesmo tempo.

        Parâmetros:
            - f, *funcoes_adicionais: funções callable f(x) (equações, derivadas, integrais...)
            - a, b: limites (se a > b, a integral sai com o sinal invertido, como na definição)
            - num_amostras: pontos sorteados em [a, b] (argumento nomeado)
            - condicoes: condições de contorno (veja Condicao); o domínio vira [a, b] ∩ condições
            - combinar: função opcional que une os valores de todas as funções num só integrando
            - semente: int/None/numpy Generator para reprodutibilidade
            - invalidos: 'ignorar' (descarta e avisa) ou 'erro'

        Retorna dict com 'integral', 'erro_padrao', 'intervalo_confianca_95' (da função única,
        do 'combinar' ou listas por função), além de 'por_funcao', 'volume_dominio',
        'amostras', 'avisos' e 'limitacoes'.
        """
        funcoes = _validar_funcoes(f, funcoes_adicionais)
        return self._integrar_n_dimensoes(funcoes, [(a, b)], num_amostras, condicoes,
                                          combinar, invalidos, semente)

    @_retorna_mensagem_de_erro
    def MonteCarloIntegracaoMultiD(self, f, limites, *funcoes_adicionais, num_amostras=100000,
                                   condicoes=(), combinar=None, semente=None, invalidos="ignorar"):
        """
        Integral de funções de várias variáveis f(*coords) por Monte Carlo sobre um domínio
        arbitrário: a caixa 'limites' intersectada com as 'condicoes' (rejeição, com estimador
        não enviesado sobre todas as tentativas).

        Parâmetros:
            - f, *funcoes_adicionais: callables f(x, y, ...) avaliados nos mesmos pontos
            - limites: [(a1, b1), (a2, b2), ...], um par por dimensão
            - num_amostras, condicoes, combinar, semente, invalidos: como em MonteCarloIntegracao1D
              (condições de igualdade são recusadas: têm medida nula)

        Também retorna 'volume_dominio' (volume do domínio admissível estimado) e 'fracao_aceita'.
        """
        funcoes = _validar_funcoes(f, funcoes_adicionais)
        return self._integrar_n_dimensoes(funcoes, limites, num_amostras, condicoes,
                                          combinar, invalidos, semente)

    # ------------------------------------------------------------------ otimização
    @_retorna_mensagem_de_erro
    def MonteCarloOtimizacao(self, f, limites, *funcoes_adicionais, num_amostras=50000,
                             minimizar=True, condicoes=(), combinar=None, semente=None,
                             invalidos="ignorar"):
        """
        Busca randômica do ótimo de funções f(*coords) em [limites] ∩ condições.

        Parâmetros:
            - f, *funcoes_adicionais: objetivos escalares reais. Com 'combinar' otimiza-se o
              escalar combinado; sem 'combinar' e com várias funções, cada uma é otimizada
              separadamente nas mesmas amostras ('por_funcao'); não há ótimo único nesse caso
              (ponto_otimo/valor_otimo vêm None) e nenhuma frente de Pareto é calculada.
            - limites: [(a1, b1), ...]
            - condicoes: restrições (desigualdade, domínio e igualdade com tolerância). Igualdades
              exigem uma amostra cair na faixa |g - valor| <= tolerancia, o que é raro: confira
              'amostras_viaveis'.
            - minimizar: True para mínimo, False para máximo
            - num_amostras, semente, invalidos: como nos métodos de integração

        Retorna 'ponto_otimo', 'valor_otimo' (±inf se nenhuma amostra viável), 'amostras_testadas',
        'amostras_viaveis', 'amostras_invalidas', 'por_funcao', 'combinado', 'avisos', 'limitacoes'.
        """
        funcoes = _validar_funcoes(f, funcoes_adicionais)
        pares = _validar_limites(limites)
        num_amostras = _validar_inteiro(num_amostras, "o número de amostras", 1)
        _validar_politica(invalidos)
        _validar_combinar(combinar)
        if not isinstance(minimizar, (bool, np.bool_)):
            raise _ErroSimulacao("'minimizar' deve ser True ou False.")
        lista_cond = _normalizar_condicoes(condicoes, permitir_igualdade=True)

        inferior = np.array([min(a, b) for a, b in pares])
        superior = np.array([max(a, b) for a, b in pares])
        rng = np.random.default_rng(semente)
        avisos = []
        sentido = 1.0 if minimizar else -1.0

        n_obj = len(funcoes) + (1 if combinar is not None else 0)
        melhores = [(math.inf, None) for _ in range(n_obj)]  # valor * sentido, ponto
        viaveis = invalidas = 0
        menor_violacao = (math.inf, None)

        for ponto in _blocos_de_pontos(rng, inferior, superior, num_amostras):
            if lista_cond:
                margem = _margem_minima(lista_cond, ponto)
                if margem < 0:
                    if -margem < menor_violacao[0]:
                        menor_violacao = (-margem, ponto)
                    continue
            saidas = _avaliar_funcoes(funcoes, combinar, ponto, escalar_real=True)
            if saidas is None:
                if invalidos == "erro":
                    raise _ErroSimulacao(
                        f"alguma função é indefinida ou não finita no ponto {ponto}. Use "
                        "invalidos='ignorar' para descartá-lo.")
                invalidas += 1
                continue
            viaveis += 1
            for i, s in enumerate(saidas):
                v = sentido * float(s)
                if v < melhores[i][0]:
                    melhores[i] = (v, ponto)

        def formatar(item):
            v, ponto = item
            if ponto is None:
                return {"ponto_otimo": None, "valor_otimo": math.inf if minimizar else -math.inf}
            return {"ponto_otimo": ponto, "valor_otimo": sentido * v}

        entradas = [formatar(m) for m in melhores]
        por_funcao = entradas[:len(funcoes)]
        combinado = entradas[len(funcoes)] if combinar is not None else None
        principal = combinado if combinado is not None else (por_funcao[0] if len(funcoes) == 1 else None)

        if viaveis == 0:
            msg = "nenhuma amostra viável foi encontrada."
            if menor_violacao[1] is not None:
                msg += (f" Menor violação observada: {menor_violacao[0]:.3g} em {menor_violacao[1]} "
                        "(nas unidades de cada condição).")
            _avisar(avisos, msg + " Aumente num_amostras ou revise condições/tolerâncias.")
        if invalidas:
            _avisar(avisos, f"{invalidas} amostras foram descartadas por avaliação indefinida/não finita.")

        return {
            "ponto_otimo": principal["ponto_otimo"] if principal else None,
            "valor_otimo": principal["valor_otimo"] if principal else None,
            "amostras_testadas": num_amostras,
            "amostras_viaveis": viaveis,
            "amostras_invalidas": invalidas,
            "por_funcao": por_funcao,
            "combinado": combinado,
            "avisos": avisos,
            "limitacoes": [
                "Busca aleatória não garante o ótimo global nem local: o valor encontrado é uma cota "
                "(superior para o mínimo, inferior para o máximo) cuja qualidade cresce devagar com N.",
                "Restrições são tratadas por rejeição; igualdades com tolerância pequena quase nunca "
                "são satisfeitas por amostragem uniforme.",
            ],
        }

    # ------------------------------------------------------------------ propagação de incerteza
    @_retorna_mensagem_de_erro
    def MonteCarloPropagacaoIncerteza(self, f, geradores_entradas, *funcoes_adicionais,
                                      num_amostras=10000, condicoes=(), combinar=None,
                                      invalidos="ignorar", num_resultados_retornados=50):
        """
        Propagação de incertezas / avaliação estocástica de n funções sobre as mesmas entradas.

        Parâmetros:
            - f, *funcoes_adicionais: callables f(*entradas) com saída escalar real
            - geradores_entradas: (a) lista de funções sem argumento, uma por entrada, ex.
              lambda: random.gauss(mu, sigma) — ENTRADAS INDEPENDENTES; ou (b) UMA função sem
              argumento que retorna a sequência completa de entradas (ex.: amostra de uma normal
              multivariada), o que permite entradas correlacionadas
            - condicoes: condições sobre as entradas; amostras que as violam são recusadas, ou seja,
              as estatísticas passam a ser CONDICIONAIS ao domínio admissível
            - combinar: função opcional que une os valores das funções numa grandeza
            - num_resultados_retornados: quantas primeiras amostras devolver
            - Reprodutibilidade: semeie você mesmo o gerador que usa (random.seed / np.random.seed)

        Retorna, por função: média, desvio padrão, erro padrão da média, mínimo, máximo, quantis e
        amostra inicial; mais covariância e correlação entre as saídas.
        """
        funcoes = _validar_funcoes(f, funcoes_adicionais)
        num_amostras = _validar_inteiro(num_amostras, "o número de amostras", 1)
        num_resultados_retornados = _validar_inteiro(num_resultados_retornados, "num_resultados_retornados", 0)
        _validar_politica(invalidos)
        _validar_combinar(combinar)
        lista_cond = _normalizar_condicoes(condicoes, permitir_igualdade=False)

        modo_conjunto = callable(geradores_entradas)
        if modo_conjunto:
            gerador_conjunto = geradores_entradas
        else:
            try:
                geradores = list(geradores_entradas)
            except TypeError:
                raise _ErroSimulacao(
                    "'geradores_entradas' deve ser uma lista de funções sem argumento ou uma única "
                    "função que retorna todas as entradas.") from None
            if not geradores or not all(callable(g) for g in geradores):
                raise _ErroSimulacao("'geradores_entradas' deve conter ao menos uma função sem argumento.")

        avisos = []
        linhas = []
        recusadas = invalidas = 0
        for _ in range(num_amostras):
            if modo_conjunto:
                try:
                    entradas = list(gerador_conjunto())
                except TypeError:
                    raise _ErroSimulacao("o gerador conjunto deve retornar uma sequência com as entradas.") from None
            else:
                entradas = [g() for g in geradores]
            if lista_cond and _margem_minima(lista_cond, entradas, parar_se_violada=True) < 0:
                recusadas += 1
                continue
            saidas = _avaliar_funcoes(funcoes, combinar, entradas, escalar_real=True)
            if saidas is None:
                if invalidos == "erro":
                    raise _ErroSimulacao(
                        f"alguma função é indefinida ou não finita nas entradas {entradas}. Use "
                        "invalidos='ignorar' para descartá-las.")
                invalidas += 1
                continue
            linhas.append([float(s) for s in saidas])

        if not linhas:
            raise _ErroSimulacao("nenhum resultado válido foi gerado nas simulações.")
        if invalidas:
            _avisar(avisos, f"{invalidas} cenários foram descartados por avaliação indefinida/não finita.")
        if recusadas:
            _avisar(avisos, f"{recusadas} cenários violaram as condições e foram recusados: as estatísticas "
                            "são condicionais ao domínio admissível.")

        dados = np.array(linhas)
        k = dados.shape[0]
        n_funcoes = len(funcoes)

        def estatisticas(coluna):
            d = dados[:, coluna]
            dp = float(np.std(d, ddof=1)) if k > 1 else 0.0
            quantis = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
            return {
                "media": float(np.mean(d)),
                "desvio_padrao": dp,
                "erro_padrao_media": dp / math.sqrt(k) if k > 1 else None,
                "minimo": float(np.min(d)),
                "maximo": float(np.max(d)),
                "quantis": {"5%": float(quantis[0]), "25%": float(quantis[1]), "50%": float(quantis[2]),
                            "75%": float(quantis[3]), "95%": float(quantis[4])},
                "amostra_resultados": d[:num_resultados_retornados].tolist(),
            }

        por_funcao = [estatisticas(j) for j in range(n_funcoes)]
        combinado = estatisticas(n_funcoes) if combinar is not None else None
        principal = combinado if combinado is not None else (por_funcao[0] if n_funcoes == 1 else None)

        covariancia = correlacao = None
        if dados.shape[1] >= 2 and k >= 2:
            covariancia = np.cov(dados, rowvar=False)
            with np.errstate(invalid="ignore", divide="ignore"):
                correlacao = np.corrcoef(dados, rowvar=False)

        resultado = dict(principal) if principal else {}
        resultado.update({
            "por_funcao": por_funcao,
            "combinado": combinado,
            "ordem_das_saidas": [f"funcao_{i + 1}" for i in range(n_funcoes)] + (["combinado"] if combinar else []),
            "covariancia": covariancia,
            "correlacao": correlacao,
            "amostras": {"tentativas": num_amostras, "validas": k,
                         "recusadas_pelas_condicoes": recusadas, "invalidas": invalidas},
            "avisos": avisos,
            "limitacoes": [
                "Com uma lista de geradores as entradas são tratadas como independentes; para correlação "
                "use um gerador conjunto. O erro padrão da média decai como N^(-1/2).",
                "Quantis, mínimo e máximo amostrais são estimativas: o extremo observado subestima a cauda "
                "real da distribuição de saída.",
            ],
        })
        return resultado

    # ------------------------------------------------------------------ simulação 3D
    @_retorna_mensagem_de_erro
    def MonteCarloSimulacao3D(self, *observaveis, ponto_inicial, deriva=None, difusao=None,
                              t_inicial=0.0, t_final=1.0, num_passos=1000, num_trajetorias=1000,
                              condicoes=(), contorno="absorver", contorno_dependente_do_tempo=False,
                              combinar=None, semente=None, invalidos="erro", max_reflexoes=20,
                              guardar_trajetorias=0):
        """
        Simulação Monte Carlo tridimensional de trajetórias de

            dX = a(X, t) dt + B(X, t) dW,      X em R³,  W: Wiener de dimensão m

        dentro de um domínio 3D delimitado por condições de contorno arbitrárias.

        Parâmetros:
            - *observaveis: funções o(x, y, z, t) avaliadas ao longo de cada trajetória (quaisquer
              callables numéricos: campos, equações, derivadas, integrais). Para cada uma são
              estimados (i) o valor terminal, (ii) o valor na saída (só trajetórias absorvidas) e
              (iii) a integral temporal ∫ o(X_s, s) ds (regra do trapézio) — a representação de
              Feynman–Kac de problemas de contorno: u(x0) = E[g(X_τ) + ∫₀^τ r(X_s) ds].
            - ponto_inicial: (x0, y0, z0), dentro do domínio admissível
            - deriva: None, vetor constante (3,) ou função a(x, y, z, t) -> 3 componentes
            - difusao: None (dinâmica determinística, integrada com RK4), escalar σ (B = σ·I),
              vetor (3,) (diagonal), matriz 3×m, ou função B(x, y, z, t) com qualquer desses
              formatos. Com difusao dada usa-se Euler–Maruyama (convenção de Itô)
            - t_inicial, t_final, num_passos: malha de tempo uniforme, dt = (t_final - t_inicial)/num_passos
            - num_trajetorias: número de trajetórias independentes
            - condicoes: define o domínio admissível (callables ou Condicao; chamadas como g(x, y, z),
              ou g(x, y, z, t) se contorno_dependente_do_tempo=True). Igualdades não são aceitas
            - contorno: 'absorver' (a trajetória termina na fronteira) ou 'refletir' (reflexão
              especular usando o gradiente numérico da condição violada; exige condições numéricas
              do tipo desigualdade, g >= 0, e não predicados booleanos)
            - combinar: função que une os valores dos observáveis numa grandeza escalar/array
            - semente: int/None/numpy Generator
            - invalidos: 'erro' (padrão: descartar trajetórias gera viés de sobrevivência) ou 'ignorar'
            - max_reflexoes: máximo de reflexões encadeadas por passo antes de rejeitar o passo
            - guardar_trajetorias: devolve os caminhos das primeiras n trajetórias válidas

        Retorna dict com estatísticas de saída (fração absorvida, tempos), posição final (média,
        covariância, erro padrão), 'observaveis' (por função), 'combinado', 'avisos', 'limitacoes'.
        """
        # ---- validações
        for i, o in enumerate(observaveis, start=1):
            if not callable(o):
                raise _ErroSimulacao(f"o observável nº {i} não é chamável ({o!r}); parâmetros como "
                                     "ponto_inicial, deriva, difusao etc. devem ser nomeados.")
        _validar_combinar(combinar)
        if combinar is not None and not observaveis:
            raise _ErroSimulacao("'combinar' exige ao menos um observável.")
        _validar_politica(invalidos)
        if contorno not in ("absorver", "refletir"):
            raise _ErroSimulacao("'contorno' deve ser 'absorver' ou 'refletir'.")
        num_passos = _validar_inteiro(num_passos, "num_passos", 1)
        num_trajetorias = _validar_inteiro(num_trajetorias, "num_trajetorias", 2)
        max_reflexoes = _validar_inteiro(max_reflexoes, "max_reflexoes", 1)
        guardar = min(_validar_inteiro(guardar_trajetorias, "guardar_trajetorias", 0), num_trajetorias)
        t0 = _validar_real_finito(t_inicial, "t_inicial")
        t1 = _validar_real_finito(t_final, "t_final")
        if t1 <= t0:
            raise _ErroSimulacao("t_final deve ser maior que t_inicial.")
        try:
            x0 = np.asarray(ponto_inicial, dtype=float)
        except (TypeError, ValueError):
            raise _ErroSimulacao("ponto_inicial deve ser numérico, no formato (x, y, z).") from None
        if x0.shape != (3,) or not np.all(np.isfinite(x0)):
            raise _ErroSimulacao("ponto_inicial deve ter exatamente 3 coordenadas finitas (x, y, z).")
        if deriva is None and difusao is None:
            raise _ErroSimulacao("informe 'deriva' e/ou 'difusao': sem elas nada se move.")

        deriva_const = None
        if deriva is not None and not callable(deriva):
            try:
                deriva_const = np.asarray(deriva, dtype=float)
            except (TypeError, ValueError):
                raise _ErroSimulacao("a deriva deve ser função a(x, y, z, t) ou vetor numérico (3,).") from None
            if deriva_const.shape != (3,):
                raise _ErroSimulacao("a deriva constante deve ter exatamente 3 componentes.")

        usa_ruido = difusao is not None
        B_const = None
        m = 0
        if usa_ruido:
            if callable(difusao):
                try:
                    m = _normalizar_difusao(difusao(*x0.tolist(), t0)).shape[1]
                except _EXCECOES_NUMERICAS:
                    raise _ErroSimulacao("a difusão não pôde ser avaliada no ponto inicial.") from None
            else:
                B_const = _normalizar_difusao(difusao)
                m = B_const.shape[1]

        lista_cond = _normalizar_condicoes(condicoes, permitir_igualdade=False)
        avisos = []

        def args_cond(pos_lista, t):
            return pos_lista + [t] if contorno_dependente_do_tempo else pos_lista

        def margem(pos, t, parar=True):
            return _margem_minima(lista_cond, args_cond(pos.tolist(), t), parar)

        if lista_cond:
            if margem(x0, t0, parar=False) < 0:
                raise _ErroSimulacao("o ponto inicial está fora do domínio admissível definido pelas condições.")
            if contorno == "refletir":
                for c in lista_cond:
                    if not c.avaliar(*args_cond(x0.tolist(), t0))[1]:
                        raise _ErroSimulacao(
                            "contorno='refletir' exige condições numéricas (g >= 0) para obter a normal; "
                            f"a condição '{c._rotulo()}' é um predicado booleano.")

        dt = (t1 - t0) / num_passos
        raiz_dt = math.sqrt(dt)

        # ---- núcleo numérico
        def avaliar_deriva(pos, t):
            if deriva_const is not None:
                return deriva_const
            v = np.asarray(deriva(*pos.tolist(), t), dtype=float)
            if v.shape != (3,):
                raise _ErroSimulacao("a função de deriva deve retornar exatamente 3 componentes.")
            return v

        def matriz_difusao(pos, t):
            B = _normalizar_difusao(difusao(*pos.tolist(), t))
            if B.shape[1] != m:
                raise _ErroSimulacao("a difusão mudou o número de colunas (m) entre chamadas.")
            return B

        def passo_rk4(pos, t, h):
            k1 = avaliar_deriva(pos, t)
            k2 = avaliar_deriva(pos + 0.5 * h * k1, t + 0.5 * h)
            k3 = avaliar_deriva(pos + 0.5 * h * k2, t + 0.5 * h)
            k4 = avaliar_deriva(pos + h * k3, t + h)
            return pos + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        def observar(pos, t):
            if not observaveis:
                return []
            return _avaliar_funcoes(observaveis, combinar, pos.tolist() + [t])

        def localizar_fronteira(a, ta, b, tb):
            # a é admissível, b não; assume um único cruzamento no segmento.
            lo, hi = 0.0, 1.0
            for _ in range(_ITERACOES_BISSECCAO):
                meio = 0.5 * (lo + hi)
                if margem(a + meio * (b - a), ta + meio * (tb - ta)) >= 0:
                    lo = meio
                else:
                    hi = meio
            return a + lo * (b - a), lo

        def normal_interna(p, destino, t):
            args_dest = args_cond(destino.tolist(), t)
            margens = [c.avaliar(*args_dest)[0] for c in lista_cond]
            c = lista_cond[int(np.argmin(margens))]
            h = 1e-6 * max(1.0, float(np.max(np.abs(p))))
            grad = np.zeros(3)
            for j in range(3):
                e = np.zeros(3)
                e[j] = h
                mais = c.avaliar(*args_cond((p + e).tolist(), t))[0]
                menos = c.avaliar(*args_cond((p - e).tolist(), t))[0]
                grad[j] = (mais - menos) / (2.0 * h)
            norma = float(np.linalg.norm(grad))
            if not math.isfinite(norma) or norma == 0.0:
                return None
            return grad / norma

        def refletir(pos, t, destino, t_novo):
            origem, t_o = pos, t
            for n_ref in range(1, max_reflexoes + 1):
                p_c, frac = localizar_fronteira(origem, t_o, destino, t_novo)
                normal = normal_interna(p_c, destino, t_novo)
                if normal is None:
                    return pos, False, n_ref - 1
                resto = destino - p_c
                destino = p_c + resto - 2.0 * float(np.dot(resto, normal)) * normal
                t_o = t_o + frac * (t_novo - t_o)
                origem = p_c
                if margem(destino, t_novo) >= 0:
                    return destino, True, n_ref
            return pos, False, max_reflexoes

        rng = np.random.default_rng(semente)

        def simular_uma(guardar_caminho):
            x = x0.copy()
            t = t0
            obs_ant = observar(x, t)
            if obs_ant is None:
                raise _TrajetoriaInvalida("observável indefinido no ponto inicial")
            integrais = [np.zeros(o.shape, dtype=o.dtype) for o in obs_ant]
            absorvida = False
            reflexoes = falhas = 0
            caminho = [x.copy()] if guardar_caminho else None
            if usa_ruido:
                ruido = rng.standard_normal((num_passos, m))
                if B_const is not None:
                    incrementos = (ruido @ B_const.T) * raiz_dt
            for i in range(num_passos):
                t_novo = t0 + (i + 1) * dt
                if usa_ruido:
                    if B_const is not None:
                        d_ruido = incrementos[i]
                    else:
                        d_ruido = (matriz_difusao(x, t) @ ruido[i]) * raiz_dt
                    x_prop = x + d_ruido
                    if deriva is not None:
                        x_prop = x_prop + avaliar_deriva(x, t) * dt
                else:
                    x_prop = passo_rk4(x, t, dt)
                if not np.all(np.isfinite(x_prop)):
                    raise _TrajetoriaInvalida("posição não finita")
                if lista_cond and margem(x_prop, t_novo) < 0:
                    if contorno == "absorver":
                        x_saida, frac = localizar_fronteira(x, t, x_prop, t_novo)
                        t_saida = t + frac * (t_novo - t)
                        obs_saida = observar(x_saida, t_saida)
                        if obs_saida is None:
                            raise _TrajetoriaInvalida("observável indefinido na saída")
                        for j in range(len(integrais)):
                            integrais[j] = integrais[j] + 0.5 * (obs_ant[j] + obs_saida[j]) * (t_saida - t)
                        x, t, obs_ant = x_saida, t_saida, obs_saida
                        absorvida = True
                        if caminho is not None:
                            caminho.append(x.copy())
                        break
                    x_prop, ok, n_ref = refletir(x, t, x_prop, t_novo)
                    reflexoes += n_ref
                    if not ok:
                        falhas += 1
                        x_prop = x.copy()
                obs_novo = observar(x_prop, t_novo)
                if obs_novo is None:
                    raise _TrajetoriaInvalida("observável indefinido ao longo da trajetória")
                for j in range(len(integrais)):
                    integrais[j] = integrais[j] + 0.5 * (obs_ant[j] + obs_novo[j]) * dt
                x, t, obs_ant = x_prop, t_novo, obs_novo
                if caminho is not None:
                    caminho.append(x.copy())
            return x, t, absorvida, reflexoes, falhas, obs_ant, integrais, caminho

        # ---- Monte Carlo
        n_q = len(observaveis) + (1 if combinar is not None else 0)
        acc_terminal = [_Acumulador() for _ in range(n_q)]
        acc_saida = [_Acumulador() for _ in range(n_q)]
        acc_integral = [_Acumulador() for _ in range(n_q)]
        posicoes, duracoes, tempos_saida, caminhos = [], [], [], []
        n_absorvidas = n_invalidas = total_reflexoes = total_falhas = 0

        for k in range(num_trajetorias):
            motivo = None
            try:
                saida = simular_uma(len(caminhos) < guardar)
            except _TrajetoriaInvalida as e:
                saida, motivo = None, str(e)
            except _EXCECOES_NUMERICAS as e:
                saida, motivo = None, repr(e)
            if saida is None:
                if invalidos == "erro":
                    raise _ErroSimulacao(
                        f"a trajetória {k + 1} produziu valores indefinidos ou não finitos ({motivo}). "
                        "Use invalidos='ignorar' para descartá-las (descartar gera viés de sobrevivência).")
                n_invalidas += 1
                continue
            x, t, absorvida, reflexoes, falhas, obs_final, integrais, caminho = saida
            posicoes.append(x)
            duracoes.append(t - t0)
            total_reflexoes += reflexoes
            total_falhas += falhas
            if absorvida:
                n_absorvidas += 1
                tempos_saida.append(t - t0)
            for j in range(n_q):
                acc_terminal[j].adicionar(obs_final[j])
                acc_integral[j].adicionar(integrais[j])
                if absorvida:
                    acc_saida[j].adicionar(obs_final[j])
            if caminho is not None:
                caminhos.append(np.array(caminho))

        n = len(posicoes)
        if n < 2:
            raise _ErroSimulacao("restaram menos de 2 trajetórias válidas.")
        if n_invalidas:
            _avisar(avisos, f"{n_invalidas} de {num_trajetorias} trajetórias foram descartadas; isso pode "
                            "enviesar as estatísticas (viés de sobrevivência).")
        if total_falhas:
            _avisar(avisos, f"{total_falhas} passos tiveram a reflexão rejeitada (normal indefinida ou "
                            f"mais de {max_reflexoes} reflexões); nesses passos a partícula ficou parada.")

        P = np.array(posicoes)
        dur = np.array(duracoes)
        p_abs = n_absorvidas / n

        def resumo_vetorial(a):
            dp = a.std(axis=0, ddof=1)
            return {"media": a.mean(axis=0), "desvio_padrao": dp, "erro_padrao": dp / math.sqrt(len(a)),
                    "n": len(a)}

        def resumo_escalar(a):
            if len(a) == 0:
                return None
            dp = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
            return {"media": float(np.mean(a)), "desvio_padrao": dp,
                    "erro_padrao": dp / math.sqrt(len(a)) if len(a) > 1 else None, "n": len(a)}

        def bloco_observavel(j):
            return {"valor_terminal": acc_terminal[j].resumo(),
                    "valor_na_saida": acc_saida[j].resumo() if acc_saida[j].n else None,
                    "integral_temporal": acc_integral[j].resumo()}

        observaveis_res = [bloco_observavel(j) for j in range(len(observaveis))]
        combinado = bloco_observavel(len(observaveis)) if combinar is not None else None

        resultado = {
            "num_trajetorias": n,
            "trajetorias_invalidas": n_invalidas,
            "passo_tempo": dt,
            "t_inicial": t0,
            "t_final": t1,
            "contorno": contorno if lista_cond else None,
            "fracao_absorvida": p_abs,
            "erro_padrao_fracao_absorvida": math.sqrt(p_abs * (1 - p_abs) / (n - 1)),
            "fracao_que_chegou_a_t_final": 1.0 - p_abs,
            "tempo_de_saida": resumo_escalar(np.array(tempos_saida)),
            "tempo_ate_parar": resumo_escalar(dur),
            "posicao_final": {**resumo_vetorial(P), "covariancia": np.cov(P, rowvar=False)},
            "reflexoes_por_trajetoria": total_reflexoes / n,
            "passos_com_reflexao_rejeitada": total_falhas,
            "observaveis": observaveis_res,
            "combinado": combinado,
            "trajetorias": caminhos if guardar else None,
            "avisos": avisos,
            "limitacoes": [
                "Euler–Maruyama (Itô) com passo fixo: ordem fraca 1 e forte 1/2 para coeficientes suaves "
                "e sem fronteiras. Se B depende de X, o resultado é a solução de Itô, não de Stratonovich.",
                "Fronteiras: a violação só é verificada ao fim de cada passo; um cruzamento seguido de "
                "retorno dentro do mesmo passo não é detectado, o que superestima a sobrevivência (viés "
                "tipicamente O(√dt) na saída). O ponto de saída/reflexão é achado por bissecção no segmento "
                "do passo (aproximação linear) assumindo um único cruzamento. Verifique reduzindo passo_tempo.",
                "A integral temporal usa a regra do trapézio sobre a malha discreta da trajetória.",
                "Com difusao=None a dinâmica é determinística (RK4, ordem 4 apenas em trechos suaves).",
            ],
        }
        return resultado


    # ------------------------------------------------------------------ otimizador inteligente
    @_retorna_mensagem_de_erro
    def OtimizadorInteligente(self, f, espaco, *funcoes_adicionais, alvo="minimizar", pesos=None,
                              combinar=None, condicoes=(), fixos=None, chamada="auto",
                              max_avaliacoes=5000, tamanho_populacao=None, paciencia=40,
                              max_reinicios=6, confirmacoes=2, tolerancia_alvo=1e-9,
                              valores_iniciais=None,
                              repeticoes=1, cache=True, num_melhores=5, semente=None,
                              invalidos="ignorar", guardar_historico=False):
        """
        Otimizador inteligente genérico: ajusta os PARÂMETROS de n cálculos (funções) para atingir um
        alvo, sobre um espaço de busca que pode misturar números, escolhas, cadeias (aminoácidos,
        DNA...), ordens, vetores e composições. Algoritmo: evolutivo híbrido (mistura de operadores DE,
        cruzamento, mutação com passo auto-adaptado e busca local; a escolha do operador tem 80% de peso
        uniforme e 20% guiado pelo sucesso recente — nos testes o aprendizado de operadores rendeu pouco
        além do sorteio uniforme), regras de Deb para restrições e reinícios independentes crescentes.
        Não usa gradientes: as funções podem ser qualquer callable (equações, derivadas, integrais,
        outros métodos do OMEGA, simulações Monte Carlo, simuladores externos).

        Parâmetros:
            - f, *funcoes_adicionais: os cálculos. Cada um recebe os parâmetros candidatos e devolve um
              número real (ou array real). Com mais de um, todos são avaliados no mesmo candidato.
            - espaco: dicionário {nome: especificação}. A especificação é um objeto Parametro
              (real, inteiro, categorico, binario, sequencia, permutacao, vetor, composicao — veja
              a classe) ou um atalho: tupla (min, max) = real (sempre, mesmo com inteiros);
              range(a, b + 1) = inteiro; lista = escolha entre opções. Também aceita uma lista de
              tuplas [(a1, b1), ...] como 'limites' dos demais métodos (nomes x1, x2, ...).
            - alvo: o que otimizar, por grandeza. 'minimizar' (padrão), 'maximizar', ou um número/vetor
              a ATINGIR (minimiza o erro quadrático médio até o alvo; com vetor, ajusta a saída da
              função a dados). Com várias grandezas: um valor único vale para todas, ou uma lista com
              uma entrada por grandeza, ex.: alvo=[2.0, "minimizar", [1, 2, 3]].
            - pesos: pesos positivos por grandeza na soma ponderada (padrão 1). Atenção: com várias
              grandezas o problema é ESCALARIZADO (soma ponderada), sem frente de Pareto; escalas muito
              diferentes pedem pesos ajustados.
            - combinar: função opcional que une os valores das funções numa única grandeza (como nos
              demais métodos); então o alvo se refere ao valor combinado.
            - condicoes: restrições sobre os parâmetros (callables ou Condicao; desigualdade, domínio
              e igualdade com tolerância). Recebem os parâmetros do mesmo modo que as funções.
            - fixos: dicionário de constantes/dados repassados às funções (não são otimizados), ex.
              fixos={"dados": y_medido}. Só com chamada 'nomeado' ou 'dicionario'.
            - chamada: como os parâmetros chegam às funções.
                'nomeado'    f(**parametros) — só os nomes que a função aceita (padrão se os nomes casam)
                'posicional' f(*valores) na ordem do dicionário 'espaco'
                'dicionario' f({nome: valor, ...})
                'auto' (padrão) decide por função: nomes que casam -> 'nomeado'; senão tantos argumentos
                posicionais quanto parâmetros -> 'posicional'; senão um único argumento -> 'dicionario'.
                O modo escolhido para cada função vem em 'chamada_utilizada': confira.
            - max_avaliacoes: orçamento de CANDIDATOS avaliados (cache não conta). Cada candidato custa
              'repeticoes' chamadas a cada função e 1 chamada a cada condição. Padrão 5000. Medido: em
              problemas triviais o critério de confirmação (abaixo) gasta ~10x mais que a primeira
              convergência (ex.: ~2000 avaliações em vez de ~200 num quadrático 2D); com cálculos
              caros use confirmacoes=1.
            - tamanho_populacao: padrão adaptado à dimensão do espaço (16 a 80).
            - paciencia: gerações sem melhora significativa que encerram uma FASE de busca. Ao fim de
              uma fase o melhor global é guardado e uma nova fase independente recomeça do zero com
              população 1,5x maior (reinícios estilo IPOP, até 'max_reinicios' vezes), o que ajuda em
              problemas com muitos mínimos locais.
            - confirmacoes: a busca termina antes do orçamento se o MESMO ótimo (tolerância relativa
              1e-6) for reencontrado por esse número de fases independentes (padrão 2). É evidência
              de convergência, não prova: 'fases' mostra o resultado de cada fase. Use um valor maior
              (ou max_reinicios maior) se o problema tiver muitos ótimos locais.
            - tolerancia_alvo: com alvos de valor em TODAS as grandezas, para quando o maior desvio
              absoluto <= tolerância (e as condições estão satisfeitas).
            - valores_iniciais: dicionário (ou lista) {nome: valor} com palpites iniciais — útil para
              refinar os valores atuais dos seus cálculos. Parâmetros omitidos entram sorteados.
            - repeticoes: avalia cada candidato n vezes e usa a média das saídas (funções ruidosas,
              como Monte Carlo). Medido (ruído gaussiano σ=0,2): o efeito foi INCONSISTENTE entre
              medições (de nenhum ganho a ~5x menos erro em x), porque o 'melhor' continua sendo a
              observação mais sortuda; reavalie sempre o resultado com mais amostras. cache: reaproveita candidatos idênticos
              (desligue com funções estocásticas se não quiser congelar o ruído da primeira avaliação).
            - num_melhores: quantas soluções distintas devolver em 'melhores_distintos'.
            - semente: int/None/Generator. Reprodutível se as funções forem determinísticas (se elas
              usam Monte Carlo, fixe a semente DENTRO delas).
            - invalidos: 'ignorar' (descarta candidatos com avaliação indefinida) ou 'erro'.
            - guardar_historico: devolve a curva de convergência por geração.

        Retorna dict com 'parametros_otimos' (exatamente o que as funções receberam; vetores e
        composições como ndarray), 'valor_otimo' (valor da grandeza única; None com várias), 'perda'
        (escalarização minimizada: +valor se minimizar, -valor se maximizar, erro quadrático médio se
        o alvo é um valor), 'viavel', 'violacao', 'alvo_atingido' (None se o alvo é min/max),
        'valores_das_funcoes', 'valor_combinado', 'desvios_do_alvo', 'melhores_distintos',
        'avaliacoes', 'geracoes', 'reinicios', 'fases', 'confirmacoes_obtidas', 'motivo_parada',
        'operadores', 'chamada_utilizada', 'configuracao', 'historico', 'avisos' e 'limitacoes'.
        """
        # ---- validações
        funcoes = _validar_funcoes(f, funcoes_adicionais)
        nomes, parametros = _normalizar_espaco(espaco)
        genes = [_criar_gene(p) for p in parametros]
        max_avaliacoes = _validar_inteiro(max_avaliacoes, "max_avaliacoes", 10)
        if tamanho_populacao is not None:
            tamanho_populacao = _validar_inteiro(tamanho_populacao, "tamanho_populacao", 5)
        paciencia = _validar_inteiro(paciencia, "paciencia", 5)
        max_reinicios = _validar_inteiro(max_reinicios, "max_reinicios", 0)
        confirmacoes = _validar_inteiro(confirmacoes, "confirmacoes", 1)
        repeticoes = _validar_inteiro(repeticoes, "repeticoes", 1)
        num_melhores = _validar_inteiro(num_melhores, "num_melhores", 1)
        tolerancia_alvo = _validar_real_finito(tolerancia_alvo, "tolerancia_alvo")
        if tolerancia_alvo < 0:
            raise _ErroSimulacao("tolerancia_alvo deve ser >= 0.")
        for nome_opcao, valor_opcao in (("cache", cache), ("guardar_historico", guardar_historico)):
            if not isinstance(valor_opcao, (bool, np.bool_)):
                raise _ErroSimulacao(f"'{nome_opcao}' deve ser True ou False.")
        _validar_politica(invalidos)
        _validar_combinar(combinar)
        if chamada not in _CHAMADAS:
            raise _ErroSimulacao("'chamada' deve ser 'auto', 'nomeado', 'posicional' ou 'dicionario'.")
        if fixos is None:
            fixos = {}
        else:
            try:
                fixos = dict(fixos)
            except (TypeError, ValueError):
                raise _ErroSimulacao("'fixos' deve ser um dicionário {nome: valor}.") from None
        if not all(isinstance(k, str) for k in fixos):
            raise _ErroSimulacao("as chaves de 'fixos' devem ser textos.")
        repetidos = [k for k in fixos if k in nomes]
        if repetidos:
            raise _ErroSimulacao(f"os nomes {repetidos} estão ao mesmo tempo em 'espaco' e em 'fixos'.")

        # ---- como as funções e condições recebem os parâmetros
        chamadores, modos, usados = [], [], set()
        for i, fn in enumerate(funcoes, start=1):
            chamar, modo, enviados = _construir_chamador(fn, chamada, nomes, fixos, f"a função nº {i}")
            chamadores.append(chamar)
            modos.append(modo)
            usados |= enviados
        lista_cond = _normalizar_condicoes(condicoes, permitir_igualdade=True)
        condicoes_adaptadas, modos_cond = [], []
        for c in lista_cond:
            rotulo = c._rotulo()
            chamar, modo, enviados = _construir_chamador(c.funcao, chamada, nomes, fixos,
                                                         f"a condição '{rotulo}'")
            condicoes_adaptadas.append(replace(c, funcao=chamar, nome=rotulo))
            modos_cond.append(modo)
            usados |= enviados

        # ---- alvo
        n_quant = 1 if combinar is not None else len(funcoes)
        rotulos = ["combinado"] if combinar is not None else [f"funcao_{i}" for i in range(1, n_quant + 1)]
        termos = _normalizar_alvo(alvo, n_quant)
        pesos_lista = _normalizar_pesos(pesos, n_quant)

        rng = np.random.default_rng(semente)
        iniciais = _codificar_iniciais(valores_iniciais, nomes, genes, rng)
        avisos = []

        dimensao = sum(g.dim for g in genes)
        if tamanho_populacao is None:
            N = int(min(80, max(16, round(8 + 6 * math.sqrt(dimensao)))))
            N = max(5, min(N, max_avaliacoes // 2))
        else:
            N = tamanho_populacao
            if N > max_avaliacoes:
                _avisar(avisos, f"tamanho_populacao ({N}) maior que max_avaliacoes ({max_avaliacoes}): "
                                "a população inicial não será avaliada por completo.")
        N = max(N, len(iniciais))

        nao_usados = [n for n in nomes if n not in usados]
        if nao_usados:
            _avisar(avisos, f"os parâmetros {nao_usados} não são recebidos por nenhuma função nem "
                            "condição: otimizá-los não tem efeito.")
        if any(c.tipo == "igualdade" for c in lista_cond):
            _avisar(avisos, "há condições de igualdade: 'viavel' = True indica só que a faixa de tolerância "
                            "foi atingida, NÃO que o ponto seja ótimo sobre a superfície de igualdade (a "
                            "busca se move mal ao longo dela). Se possível, reparametrize para eliminá-la.")
        if n_quant > 1:
            _avisar(avisos, f"{n_quant} grandezas foram escalarizadas por soma ponderada (pesos={pesos_lista}); "
                            "o resultado depende das escalas e dos pesos e não é uma frente de Pareto.")

        avaliador = _Avaliador(genes, nomes, chamadores, combinar, termos, pesos_lista,
                               condicoes_adaptadas, invalidos, repeticoes, bool(cache), tolerancia_alvo)
        motor = _MotorEvolutivo(genes, avaliador, rng, max_avaliacoes, N, paciencia, max_reinicios,
                                confirmacoes, iniciais, num_melhores, bool(guardar_historico))
        motor.executar()

        # ---- resultado
        genoma, reg, payload = motor.melhor
        params = avaliador.parametros(genoma)
        viavel = reg[0] == 0.0
        if payload is None and reg[0] != math.inf:
            payload = avaliador.valores(params)  # melhor inviável: avaliado só para informar
        saidas, combinado = payload if payload is not None else (None, None)

        desvios, valor_otimo = [], None
        if saidas is not None:
            itens = avaliador.contribuicoes(saidas, combinado)
            for rotulo, (tipo, _), item in zip(rotulos, termos, itens):
                if tipo == "valor":
                    desvios.append({"grandeza": rotulo, "erro_absoluto_maximo": item[1],
                                    "erro_quadratico_medio": item[2]})
            if n_quant == 1:
                valor_otimo = _saida(combinado if combinar is not None else saidas[0])

        if not viavel:
            if reg[0] == math.inf:
                _avisar(avisos, "todas as avaliações foram indefinidas/não finitas; revise o espaço de "
                                "busca e as funções.")
            else:
                _avisar(avisos, f"nenhum candidato viável foi encontrado; devolvendo o de menor violação "
                                f"({reg[0]:.3g}, nas unidades de cada condição). Aumente max_avaliacoes ou "
                                "revise condições/tolerâncias (igualdades exigem tolerância folgada).")
        if avaliador.sem_gradiente and not viavel:
            _avisar(avisos, "há condições booleanas (predicados): elas não indicam o quão perto da "
                            "fronteira o candidato está, então a busca por viabilidade é praticamente cega.")
        if motor.n_indefinidas:
            _avisar(avisos, f"{motor.n_indefinidas} avaliações foram descartadas por resultado "
                            "indefinido/não finito.")
        if motor.motivo == "orçamento de avaliações esgotado":
            if motor.sem_melhora_final < 3:
                _avisar(avisos, "o orçamento acabou enquanto o algoritmo ainda estava melhorando; a "
                                "solução provavelmente não convergiu — aumente max_avaliacoes.")
            elif viavel and motor.confirmadas < confirmacoes:
                _avisar(avisos, f"o melhor valor foi reencontrado por {motor.confirmadas} fase(s) "
                                f"independente(s) (pedidas: {confirmacoes}); trate-o como candidato, não "
                                "como ótimo estabelecido, ou aumente max_avaliacoes.")

        melhores = []
        for reg_e, gen_e, pay_e in sorted(motor.elite.values(), key=lambda t: (t[0][0], t[0][1])):
            s_e, c_e = pay_e
            melhores.append({"parametros": avaliador.parametros(gen_e), "perda": reg_e[1],
                             "valores_das_funcoes": [_saida(a) for a in s_e],
                             "valor_combinado": None if c_e is None else _saida(c_e)})

        probabilidades = motor.probabilidades()
        operadores = {nome: {"usos": motor.usos[k], "melhorias_sobre_a_base": motor.sucessos[k],
                             "novos_melhores_globais": motor.novos_melhores[k],
                             "probabilidade_final": probabilidades[k]}
                      for k, nome in enumerate(motor.OPERADORES)}

        return {
            "parametros_otimos": params,
            "valor_otimo": valor_otimo,
            "perda": reg[1] if math.isfinite(reg[1]) else None,
            "viavel": viavel,
            "violacao": reg[0] if math.isfinite(reg[0]) else None,
            "alvo_atingido": reg[2] if viavel else None,
            "alvo": [{"grandeza": r, "tipo": t[0], "valor_alvo": None if t[1] is None else _saida(t[1]),
                      "peso": w} for r, t, w in zip(rotulos, termos, pesos_lista)],
            "valores_das_funcoes": None if saidas is None else [_saida(a) for a in saidas],
            "valor_combinado": None if combinado is None else _saida(combinado),
            "desvios_do_alvo": desvios,
            "melhores_distintos": melhores,
            "avaliacoes": {"executadas": motor.n_aval, "reaproveitadas_do_cache": motor.n_cache,
                           "indefinidas": motor.n_indefinidas, "inviaveis": motor.n_inviaveis},
            "geracoes": motor.geracoes,
            "reinicios": motor.reinicios,
            "fases": motor.fases,
            "confirmacoes_obtidas": motor.confirmadas,
            "motivo_parada": motor.motivo,
            "geracoes_sem_melhora_ao_final": motor.sem_melhora_final,
            "diversidade_final": motor.diversidade_final,
            "operadores": operadores,
            "chamada_utilizada": {"funcoes": modos, "condicoes": modos_cond},
            "configuracao": {"tamanho_populacao": N, "dimensao_efetiva": dimensao,
                             "max_avaliacoes": max_avaliacoes, "paciencia": paciencia,
                             "repeticoes": repeticoes, "cache": bool(cache)},
            "historico": motor.historico,
            "avisos": avisos,
            "limitacoes": [
                "Metaheurística estocástica: não garante o ótimo global nem prova de otimalidade; resultados "
                "diferem entre sementes. Para confiar num resultado, repita com sementes diferentes e "
                "compare 'perda' e 'parametros_otimos'.",
                "Não usa gradientes nem modelo substituto: o custo cresce com a dimensão e com o orçamento "
                "de avaliações das suas funções. Espaços grandes e funções caras pedem max_avaliacoes alto.",
                "Várias grandezas viram uma soma ponderada (escalarização); não há frente de Pareto e o "
                "ótimo depende dos pesos e das escalas.",
                "Restrições seguem as regras de Deb. IGUALDADES são o ponto fraco: achar um ponto viável "
                "na faixa |g - valor| <= tolerância é possível, mas a busca quase não consegue se mover ao "
                "longo dela. Medido em min x s.a. x²+y²=1 (ótimo x=-1): tolerância 1e-4 devolveu x=+0,66 "
                "marcado como viável; reparametrizando (x=cos θ, y=sin θ) o resultado foi exato. Prefira "
                "eliminar a igualdade (reparametrização, Parametro.composicao para somas fixas). Predicados "
                "booleanos (sem distância à fronteira) também são difíceis de satisfazer.",
                "Se as funções são ruidosas (ex.: Monte Carlo), o 'melhor' é a observação mais sortuda; "
                "o otimizador ajusta exatamente o que a função devolve, inclusive o erro dela (ex.: um "
                "integral Monte Carlo com erro padrão 2e-3 leva a parâmetros com erro dessa ordem). Use "
                "repeticoes > 1 e/ou cache=False e reavalie o resultado com mais amostras.",
                "A aprendizagem de operadores é fraca (80% do peso é uniforme); em testes comparáveis, "
                "hill-climb com reinícios empatou ou superou este método em paisagens discretas "
                "epistáticas, e o DE do scipy foi melhor em Rosenbrock com orçamento pequeno. Para "
                "problemas contínuos suaves, métodos com gradiente (scipy.optimize) usam ordens de "
                "grandeza menos avaliações.",
            ],
        }