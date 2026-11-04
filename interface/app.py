"""OMEGA Utils — Interface Web amigável (Flask).

Painel visual sobre os módulos da biblioteca: integrais, derivadas, equações,
funções, matrizes, simulações, estruturas, materiais, constantes, elementos e
a demonstração de escoamento em bocal Venturi.

Executar a partir da raiz do projeto:

    python interface/app.py            # http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

import numpy as np
from flask import Flask, jsonify, render_template, request

from interface.matheval import (
    ErroExpressao,
    criar_funcao,
    formatar_numero,
    matriz,
    numero,
    vetor,
)

# ---------------------------------------------------------------------------
# Solvers do OMEGA Utils (imports tolerantes a dependências ausentes)
# ---------------------------------------------------------------------------

def _importar_seguro(tipo_erro, mensagem):
    print(f"[aviso] {mensagem}", file=sys.stderr)
    return None


try:
    from constantes.constantes import Constantes
except Exception:  # pragma: no cover
    Constantes = None

try:
    from derivadas.derivadas import DerivadasSolver
except Exception:  # pragma: no cover
    DerivadasSolver = None

try:
    from equacoes.equacoes import EquacoesSolver, EDOsSolver, TeoremasSolver
except Exception:  # pragma: no cover
    EquacoesSolver = EDOsSolver = TeoremasSolver = None

try:
    from funcoes.funcoes import FuncoesSolver, FuncoesSolverNaoLinear
except Exception:  # pragma: no cover
    FuncoesSolver = FuncoesSolverNaoLinear = None

try:
    from integrais.integrais import IntegralSolverDefinidas, IntegralSolverIndefinidas
except Exception:  # pragma: no cover
    IntegralSolverDefinidas = IntegralSolverIndefinidas = None

try:
    from simulacoes.simulacoes import SimulacoesSolver
except Exception:  # pragma: no cover
    SimulacoesSolver = None

try:
    from estruturas.estruturas import AnaliseEstruturalSolver
except Exception:  # pragma: no cover
    AnaliseEstruturalSolver = None

try:
    from materiais.materiais import Lamina, LaminaOrtotropica, MateriaisCompositosSolver
except Exception:  # pragma: no cover
    Lamina = LaminaOrtotropica = MateriaisCompositosSolver = None

try:
    from Elementos_Materiais.EMF import ElementosMateriais
except Exception:  # pragma: no cover
    ElementosMateriais = None

try:
    from main.main import SimulacaoTempoRealVenturi3D
except Exception:  # pragma: no cover
    SimulacaoTempoRealVenturi3D = None

_MATRIZES_DISPONIVEIS = False
MatrizesSolver = None
try:
    import tensorflow  # noqa: F401  (usado internamente pelo módulo)

    from matrizes.matrizes import MatrizesSolver

    _MATRIZES_DISPONIVEIS = True
except Exception:
    _MATRIZES_DISPONIVEIS = False

CONST = Constantes() if Constantes else None
DERIVADAS = DerivadasSolver() if DerivadasSolver else None
EQS = EquacoesSolver() if EquacoesSolver else None
EDO = EDOsSolver() if EDOsSolver else None
TEOREMAS = TeoremasSolver() if TeoremasSolver else None
FUNCOES = FuncoesSolver() if FuncoesSolver else None
NAO_LINEAR = FuncoesSolverNaoLinear() if FuncoesSolverNaoLinear else None
INT_DEF = IntegralSolverDefinidas() if IntegralSolverDefinidas else None
SIMULACOES = SimulacoesSolver() if SimulacoesSolver else None
ESTRUTURAS = AnaliseEstruturalSolver() if AnaliseEstruturalSolver else None
MATERIAIS = MateriaisCompositosSolver() if MateriaisCompositosSolver else None
MATRIZES = MatrizesSolver() if _MATRIZES_DISPONIVEIS else None

MAX_PLOT_PONTOS = 400


def _funcao_x(texto):
    """Compila expressão em x; retorna (f, rotulo). f(x)->float."""
    func, rotulo = criar_funcao(texto, variaveis=("x",))
    return (lambda x: func(x=x)), rotulo


def _numpy_para_lista(arr):
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=np.inf * 0.0, posinf=1e308, neginf=-1e308)
    return [None if not np.isfinite(v) else float(v) for v in arr.ravel()]


def _curva_de_funcao(f, a, b, n=MAX_PLOT_PONTOS):
    xs = np.linspace(a, b, n)
    ys = []
    for xi in xs:
        try:
            ys.append(float(f(float(xi))))
        except Exception:
            ys.append(None)
    return _numpy_para_lista(xs), ys


def _resposta_ok(**dados):
    payload = {"ok": True}
    payload.update(dados)
    return jsonify(payload)


def _resposta_erro(mensagem, status=400):
    return jsonify({"ok": False, "erro": str(mensagem)}), status


def _tratar(funcao):
    """Executa `funcao()` convertendo erros em respostas amigáveis."""
    try:
        return funcao()
    except ErroExpressao as exc:
        return _resposta_erro(exc)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        return _resposta_erro(exc)
    except Exception as exc:  # erro interno inesperado
        traceback.print_exc()
        return _resposta_erro(f"Erro interno: {exc}", status=500)


# ---------------------------------------------------------------------------
# App e página principal
# ---------------------------------------------------------------------------

app = Flask(__name__)


def _modulo_disponivel(nome, solver):
    return {"nome": nome, "disponivel": solver is not None}


@app.route("/")
def index():
    modulos = [
        ("integrais", "Integrais", INT_DEF is not None or SIMULACOES is not None, "calc"),
        ("derivadas", "Derivadas", DERIVADAS is not None, "derivative"),
        ("equacoes", "Equações & Física", EQS is not None or EDO is not None, "equation"),
        ("funcoes", "Funções & Raízes", FUNCOES is not None or NAO_LINEAR is not None, "root"),
        ("teoremas", "Teoremas Vetoriais", TEOREMAS is not None, "theorem"),
        ("montecarlo", "Simulação Monte Carlo", SIMULACOES is not None, "dice"),
        ("estruturas", "Análise Estrutural", ESTRUTURAS is not None, "structure"),
        ("materiais", "Materiais Compósitos", MATERIAIS is not None, "layers"),
        ("matrizes", "Álgebra Linear (Matrizes)", MATRIZES is not None, "matrix"),
        ("constantes", "Constantes", CONST is not None, "atom"),
        ("elementos", "Tabela Periódica & Fluidos", ElementosMateriais is not None, "table"),
        ("venturi", "Escoamento Venturi 3D", SimulacaoTempoRealVenturi3D is not None, "wind"),
    ]
    avisos = []
    if MATRIZES is None:
        avisos.append(
            "O módulo de Matrizes requer TensorFlow (pip install tensorflow) — "
            "as demais ferramentas seguem disponíveis."
        )
    return render_template("index.html", modulos=modulos, avisos=avisos)


# ---------------------------------------------------------------------------
# API — Integrais
# ---------------------------------------------------------------------------

@app.route("/api/integral/numerica", methods=["POST"])
def api_integral_numerica():
    def executar():
        dados = request.get_json(force=True)
        if INT_DEF is None:
            raise ErroExpressao("módulo de integrais indisponível neste ambiente.")
        f, rotulo = _funcao_x(dados.get("funcao", ""))
        a = numero(dados.get("a"), "limite inferior")
        b = numero(dados.get("b"), "limite superior")
        metodo = (dados.get("metodo") or "simpson").lower()
        n = int(numero(dados.get("n", 1000), "subintervalos"))
        if b == a:
            raise ErroExpressao("os limites de integração devem ser diferentes.")

        if metodo == "simpson":
            if n % 2:
                n += 1
            valor = INT_DEF.RegraDeSimpson(f, a, b, n)
            detalhe = f"Simpson 1/3 composto com {n} subintervalos"
        elif metodo == "trapezio":
            tol = float(dados.get("tol") or 1e-8)
            valor = INT_DEF.RegraDoTrapezioComposta(a, b, f, max(n, 2), tol)
            detalhe = f"Trapézio composto (tolerância {tol:g})"
        elif metodo == "adaptativa":
            tol = float(dados.get("tol") or 1e-9)
            resultado = INT_DEF.IntegracaoSimpsonAdaptativa(f, a, b, tolerancia=tol)
            if isinstance(resultado, dict):
                valor = resultado.get("valor", resultado.get("integral"))
                detalhe = f"Simpson adaptativa (tol {tol:g}) — {resultado}"
            elif isinstance(resultado, (list, tuple)):
                valor = resultado[0]
                detalhe = f"Simpson adaptativa (tol {tol:g})"
            else:
                valor = resultado
                detalhe = f"Simpson adaptativa (tol {tol:g})"
        else:
            raise ErroExpressao("método desconhecido; use simpson, trapezio ou adaptativa.")

        xs, ys = _curva_de_funcao(f, a, b)
        return _resposta_ok(
            valor=formatar_numero(valor),
            valor_bruto=float(valor),
            detalhe=detalhe,
            expressao=rotulo,
            grafico={"x": xs, "y": ys, "a": a, "b": b},
        )

    return _tratar(executar)


@app.route("/api/integral/linha", methods=["POST"])
def api_integral_linha():
    def executar():
        dados = request.get_json(force=True)
        if IntegralSolverIndefinidas is None:
            raise ErroExpressao("módulo de integrais indisponível neste ambiente.")
        fx, _ = criar_funcao(dados.get("px", ""), variaveis=("t",))
        fy, _ = criar_funcao(dados.get("py", ""), variaveis=("t",))
        cx, rx = criar_funcao(dados.get("cx", ""), variaveis=("t",))
        cy, ry = criar_funcao(dados.get("cy", ""), variaveis=("t",))
        t0 = numero(dados.get("t0"), "t inicial")
        t1 = numero(dados.get("t1"), "t final")

        campo = lambda t: np.array([fx(t=t), fy(t=t)], dtype=float)
        curva = lambda t: np.array([cx(t=t), cy(t=t)], dtype=float)
        valor = IntegralSolverIndefinidas().IntegralDeLinha(campo, curva, t0, t1)

        ts = np.linspace(t0, t1, 200)
        return _resposta_ok(
            valor=formatar_numero(valor),
            detalhe="∫ F·dr ao longo da curva parametrizada r(t) = (%s, %s)" % (rx, ry),
            grafico={
                "curva_x": _numpy_para_lista(np.array([cx(t=float(t)) for t in ts])),
                "curva_y": _numpy_para_lista(np.array([cy(t=float(t)) for t in ts])),
            },
        )

    return _tratar(executar)


# ---------------------------------------------------------------------------
# API — Derivadas
# ---------------------------------------------------------------------------

@app.route("/api/derivada", methods=["POST"])
def api_derivada():
    def executar():
        dados = request.get_json(force=True)
        if DERIVADAS is None:
            raise ErroExpressao("módulo de derivadas indisponível neste ambiente.")
        f, rotulo = _funcao_x(dados.get("funcao", ""))
        x0 = numero(dados.get("ponto"), "ponto x₀")
        h = float(dados.get("h") or 1e-5)
        solver = DerivadasSolver(h=h)

        primeira = solver.Derivada(f, x0)
        segunda = solver.SegundaDerivadaCentralizada(f, x0)
        progressiva = solver.DerivadaProgressiva(f, x0)
        centralizada = solver.DerivadaCentralizada(f, x0)

        a_reta = primeira
        b_reta = f(x=x0) - primeira * x0
        xs, ys = _curva_de_funcao(f, x0 - 2.0, x0 + 2.0)
        tangente = [a_reta * x + b_reta for x in xs]

        return _resposta_ok(
            primeira=formatar_numero(primeira),
            segunda=formatar_numero(segunda),
            progressiva=formatar_numero(progressiva),
            centralizada=formatar_numero(centralizada),
            expressao=rotulo,
            tangente={"declive": formatar_numero(a_reta), "intercepto": formatar_numero(b_reta)},
            grafico={"x": xs, "y": ys, "tangente": tangente, "ponto": [x0, f(x=x0)]},
        )

    return _tratar(executar)


# ---------------------------------------------------------------------------
# API — Equações & Física
# ---------------------------------------------------------------------------

@app.route("/api/equacao/grau2", methods=["POST"])
def api_equacao_grau2():
    def executar():
        dados = request.get_json(force=True)
        if EQS is None:
            raise ErroExpressao("módulo de equações indisponível neste ambiente.")
        a = numero(dados.get("a"), "coeficiente a")
        b = numero(dados.get("b"), "coeficiente b")
        c = numero(dados.get("c"), "coeficiente c")
        raizes = EQS.EquacaoGrau2(a, b, c)
        xs = np.linspace(-6, 6, 240)
        ys = [a * x * x + b * x + c for x in xs]
        raizes_lista = []
        if isinstance(raizes, (list, tuple)):
            for r in raizes:
                try:
                    raizes_lista.append(complex(r))
                except TypeError:
                    raizes_lista.append(str(r))
        return _resposta_ok(
            bruto=raizes,
            raizes=[formatar_numero(r.real) if isinstance(r, complex) and abs(r.imag) < 1e-12
                    else (f"{formatar_numero(r.real)} {r.imag:+.4g}i" if isinstance(r, complex) else str(r))
                    for r in raizes_lista] or [str(raizes)],
            delta=formatar_numero(b * b - 4 * a * c),
            grafico={"x": _numpy_para_lista(xs), "y": _numpy_para_lista(ys)},
        )

    return _tratar(executar)


@app.route("/api/fisica/mrv", methods=["POST"])
def api_fisica_mrv():
    def executar():
        dados = request.get_json(force=True)
        if EQS is None:
            raise ErroExpressao("módulo de equações indisponível neste ambiente.")
        s0 = numero(dados.get("s0", 0), "posição inicial S₀")
        v0 = numero(dados.get("v0"), "velocidade inicial v₀")
        acc = numero(dados.get("a"), "aceleração a")
        tempo_final = numero(dados.get("t", 10), "tempo total t")
        posicao = EQS.MovimentoUniformeVariado(s0, v0, tempo_final, acc)
        ts = np.linspace(0, tempo_final, 200)
        pos = s0 + v0 * ts + 0.5 * acc * ts ** 2
        vel = v0 + acc * ts
        return _resposta_ok(
            posicao=formatar_numero(posicao),
            velocidade_final=formatar_numero(v0 + acc * tempo_final),
            detalhe=f"S(t) = S₀ + v₀t + at²/2 avaliado em t = {tempo_final:g} s",
            grafico={
                "tipo": "dupla",
                "x": _numpy_para_lista(ts),
                "serie1": _numpy_para_lista(pos),
                "serie2": _numpy_para_lista(vel),
                "rotulo1": "Posição (m)",
                "rotulo2": "Velocidade (m/s)",
            },
        )

    return _tratar(executar)


@app.route("/api/edo/rk4", methods=["POST"])
def api_edo_rk4():
    def executar():
        dados = request.get_json(force=True)
        if EDO is None:
            raise ErroExpressao("módulo de EDOs indisponível neste ambiente.")
        expr = dados.get("dydt", "").strip() or "-y"
        func, rotulo = criar_funcao(expr, variaveis=("t", "y"))
        f = lambda t, y: func(t=t, y=y)
        y0 = numero(dados.get("y0", 1), "condição inicial y(0)")
        h = numero(dados.get("h", 0.1), "passo h")
        passos = int(numero(dados.get("passos", 50), "número de passos"))
        t0 = numero(dados.get("t0", 0), "t inicial")
        tempos, estados = EDO.Rk4(f, y0=y0, h=h, passos=passos, t0=t0)
        estados = np.asarray(estados, dtype=float).ravel()
        return _resposta_ok(
            final=formatar_numero(estados[-1]),
            detalhe=f"y' = {rotulo}, y({t0:g}) = {y0:g}, passo {h:g}, {passos} passos",
            grafico={"x": _numpy_para_lista(tempos), "y": _numpy_para_lista(estados)},
        )

    return _tratar(executar)


# ---------------------------------------------------------------------------
# API — Funções & Raízes
# ---------------------------------------------------------------------------

@app.route("/api/raiz/bissecao", methods=["POST"])
def api_raiz_bissecao():
    def executar():
        dados = request.get_json(force=True)
        if NAO_LINEAR is None:
            raise ErroExpressao("módulo de funções não lineares indisponível neste ambiente.")
        f, rotulo = _funcao_x(dados.get("funcao", ""))
        a = numero(dados.get("a"), "extremo a")
        b = numero(dados.get("b"), "extremo b")
        n_iter = int(numero(dados.get("n", 60), "iterações"))
        tol = float(dados.get("tol") or 1e-8)
        raiz = NAO_LINEAR.MetodoBissecao(f, a, b, n_iter, tol)
        xs, ys = _curva_de_funcao(f, a, b)
        return _resposta_ok(
            raiz=formatar_numero(raiz),
            f_na_raiz=formatar_numero(f(float(raiz))),
            detalhe=f"Bisseção de f(x) = {rotulo} em [{a:g}, {b:g}]",
            grafico={"x": xs, "y": ys, "marca": [float(raiz), 0.0]},
        )

    return _tratar(executar)


@app.route("/api/raiz/newton", methods=["POST"])
def api_raiz_newton():
    def executar():
        dados = request.get_json(force=True)
        if NAO_LINEAR is None or DERIVADAS is None:
            raise ErroExpressao("módulos de funções/derivadas indisponíveis neste ambiente.")
        f, rotulo = _funcao_x(dados.get("funcao", ""))
        df = lambda x: DERIVADAS.DerivadaCentralizada(f, x)
        x0 = numero(dados.get("x0"), "chute inicial x₀")
        tol = float(dados.get("tol") or 1e-7)
        max_iter = int(numero(dados.get("max_iter", 100), "iterações máximas"))
        raiz = NAO_LINEAR.NewtonRaphson(f, df, x0, tol=tol, max_iter=max_iter)
        xs, ys = _curva_de_funcao(f, x0 - 3, x0 + 3)
        return _resposta_ok(
            raiz=formatar_numero(raiz),
            f_na_raiz=formatar_numero(f(float(raiz))),
            detalhe=f"Newton–Raphson de f(x) = {rotulo} a partir de x₀ = {x0:g}",
            grafico={"x": xs, "y": ys, "marca": [float(raiz), 0.0]},
        )

    return _tratar(executar)


@app.route("/api/polimonio", methods=["POST"])
def api_polimonio():
    def executar():
        dados = request.get_json(force=True)
        if FUNCOES is None:
            raise ErroExpressao("módulo de funções indisponível neste ambiente.")
        coeficientes = vetor(dados.get("coeficientes", ""), "coeficientes")
        grau = len(coeficientes) - 1
        if grau < 1 or grau > 4:
            raise ErroExpressao("informe coeficientes de um polinômio de grau 1 a 4.")
        args = coeficientes[::-1]  # a_n ... a_0 -> assinatura dos métodos
        metodos = {
            1: FUNCOES.PolinomialGrau1,
            2: FUNCOES.PolinomialGrau2,
            3: FUNCOES.PolinominalGrau3,
            4: FUNCOES.PolinomialGrau4,
        }
        raizes = metodos[grau](*args)
        poly = np.poly1d(coeficientes)
        polos = np.roots(coeficientes)
        dominio = max(5.0, 2 * max((abs(r) for r in polos), default=2.5))
        xs = np.linspace(-dominio, dominio, MAX_PLOT_PONTOS)
        ys = [poly(x) for x in xs]
        lista = []
        try:
            iteravel = raizes if isinstance(raizes, (list, tuple)) else [raizes]
            for r in iteravel:
                try:
                    cr = complex(r)
                    lista.append(formatar_numero(cr.real) if abs(cr.imag) < 1e-10
                                 else f"{formatar_numero(cr.real)} {cr.imag:+.4g}i")
                except TypeError:
                    lista.append(str(r))
        except Exception:
            lista = [str(raizes)]
        return _resposta_ok(
            raizes=lista,
            detalhe=f"Polinômio de grau {grau}: " + " ".join(
                f"+{c:g}" if c >= 0 else f"-{abs(c):g}" for c in coeficientes
            ),
            grafico={"x": _numpy_para_lista(xs), "y": _numpy_para_lista(ys)},
        )

    return _tratar(executar)


@app.route("/api/limite", methods=["POST"])
def api_limite():
    def executar():
        dados = request.get_json(force=True)
        if FUNCOES is None:
            raise ErroExpressao("módulo de funções indisponível neste ambiente.")
        f, rotulo = _funcao_x(dados.get("funcao", ""))
        x0 = numero(dados.get("ponto"), "ponto x₀")
        direcao = dados.get("direcao") or "ambos"
        valor = FUNCOES.limite_numerico(f, x0, direcao=direcao)
        xs_left = np.linspace(max(x0 - 1.0, -1e6), x0 - 1e-6, 120)
        xs_right = np.linspace(x0 + 1e-6, x0 + 1.0, 120)
        xs = np.concatenate([xs_left, xs_right])
        ys = []
        for xi in xs:
            try:
                ys.append(float(f(float(xi))))
            except Exception:
                ys.append(None)
        return _resposta_ok(
            limite=formatar_numero(valor),
            detalhe=f"lim x→{x0:g} f(x), f(x) = {rotulo}",
            grafico={"x": _numpy_para_lista(xs), "y": _numpy_para_lista(ys),
                     "marca": [x0, float(valor) if valor is not None else 0]},
        )

    return _tratar(executar)


# ---------------------------------------------------------------------------
# API — Teoremas vetoriais
# ---------------------------------------------------------------------------

@app.route("/api/teorema/green", methods=["POST"])
def api_teorema_green():
    def executar():
        dados = request.get_json(force=True)
        if TEOREMAS is None:
            raise ErroExpressao("módulo de teoremas vetoriais indisponível neste ambiente.")
        P, _ = criar_funcao(dados.get("P", "y"), variaveis=("x", "y"))
        Q, _ = criar_funcao(dados.get("Q", "-x"), variaveis=("x", "y"))
        cx, _ = criar_funcao(dados.get("cx", "cos(t)"), variaveis=("t",))
        cy, _ = criar_funcao(dados.get("cy", "sin(t)"), variaveis=("t",))
        curva = lambda t: np.array([cx(t=t), cy(t=t)])
        resultado = TEOREMAS.CalcularTeoremaDeGreen(
            lambda x, y: P(x=x, y=y),
            lambda x, y: Q(x=x, y=y),
            curva,
            (0.0, 2 * math.pi),
            ((-1.5, 1.5), (-1.5, 1.5)),
        )
        ts = np.linspace(0, 2 * math.pi, 200)
        return _resposta_ok(
            resultado={k: (formatar_numero(v) if isinstance(v, (int, float)) else str(v))
                       for k, v in resultado.items()},
            detalhe="Teorema de Green: ∮ P dx + Q dy = ∬ (∂Q/∂x − ∂P/∂y) dA",
            grafico={
                "curva_x": _numpy_para_lista(np.array([cx(t=float(t)) for t in ts])),
                "curva_y": _numpy_para_lista(np.array([cy(t=float(t)) for t in ts])),
            },
        )

    return _tratar(executar)


# ---------------------------------------------------------------------------
# API — Monte Carlo
# ---------------------------------------------------------------------------

@app.route("/api/montecarlo/integracao", methods=["POST"])
def api_montecarlo_integracao():
    def executar():
        dados = request.get_json(force=True)
        if SIMULACOES is None:
            raise ErroExpressao("módulo de simulações indisponível neste ambiente.")
        f, rotulo = _funcao_x(dados.get("funcao", ""))
        a = numero(dados.get("a"), "limite inferior")
        b = numero(dados.get("b"), "limite superior")
        amostras = int(numero(dados.get("amostras", 100000), "número de amostras"))
        semente = dados.get("semente")
        kw = {}
        if semente not in (None, ""):
            kw["semente"] = int(semente)
        resultado = SIMULACOES.MonteCarloIntegracao1D(f, a, b, num_amostras=amostras, **kw)
        if isinstance(resultado, dict):
            valor = resultado.get("integral", resultado.get("valor"))
            resumo = {k: (formatar_numero(v) if isinstance(v, (int, float)) else v)
                      for k, v in resultado.items()}
        else:
            valor = resultado[0]
            resumo = {"valor": formatar_numero(valor)}
        exato = None
        try:
            exato = INT_DEF.RegraDeSimpson(f, a, b, 2000)
        except Exception:
            pass
        xs, ys = _curva_de_funcao(f, a, b)
        return _resposta_ok(
            valor=formatar_numero(valor),
            resumo=resumo,
            referencia_simpson=formatar_numero(exato) if exato is not None else None,
            detalhe=f"Monte Carlo (amostragem uniforme) de f(x) = {rotulo} em [{a:g}, {b:g}]",
            grafico={"x": xs, "y": ys, "a": a, "b": b},
        )

    return _tratar(executar)


# ---------------------------------------------------------------------------
# API — Análise estrutural
# ---------------------------------------------------------------------------

@app.route("/api/estrutura/euler", methods=["POST"])
def api_estrutura_euler():
    def executar():
        dados = request.get_json(force=True)
        if ESTRUTURAS is None:
            raise ErroExpressao("módulo de análise estrutural indisponível neste ambiente.")
        e = numero(dados.get("modulo"), "módulo de elasticidade E") * 1e9  # GPa -> Pa
        inercia = numero(dados.get("inercia"), "momento de inércia I") * 1e-8  # cm⁴ -> m⁴
        comprimento = numero(dados.get("comprimento"), "comprimento L") / 100  # cm -> m
        k = numero(dados.get("k", 1), "fator de comprimento efetivo K")
        pcr = ESTRUTURAS.carga_critica_euler(e, inercia, comprimento, k)
        carga_kn = pcr / 1000
        tensao_admissivel = numero(dados.get("fs", 2), "fator de segurança FS")
        return _resposta_ok(
            carga_critica=formatar_numero(pcr) + " N",
            carga_critica_kn=formatar_numero(carga_kn) + " kN",
            projeto=formatar_numero(carga_kn / tensao_admissivel) + " kN",
            detalhe=(
                f"Pcr = π²EI/(KL)² com E = {e/1e9:g} GPa, I = {inercia*1e8:g} cm⁴, "
                f"L = {comprimento*100:g} cm, K = {k:g}"
            ),
        )

    return _tratar(executar)


@app.route("/api/estrutura/fratura", methods=["POST"])
def api_estrutura_fratura():
    def executar():
        dados = request.get_json(force=True)
        if ESTRUTURAS is None:
            raise ErroExpressao("módulo de análise estrutural indisponível neste ambiente.")
        sigma = numero(dados.get("tensao"), "tensão σ") * 1e6  # MPa -> Pa
        kic = numero(dados.get("kic"), "tenacidade K_IC") * 1e6  # MPa·√m → Pa·√m
        y = numero(dados.get("y", 1), "fator geométrico Y")
        a_crit = ESTRUTURAS.comprimento_critico_trinca(kic, sigma, y)
        ki = ESTRUTURAS.fator_intensidade_tensao_modo_i(sigma, a_crit / 4, y)
        return _resposta_ok(
            comprimento_critico=formatar_numero(a_crit * 1000) + " mm",
            ki_no_quarto=formatar_numero(ki / 1e6) + " MPa·√m",
            detalhe=(
                f"a_c = (K_IC/(Yσ))²/π com K_IC = {kic/1e6:g} MPa√m, σ = {sigma/1e6:g} MPa, Y = {y:g}"
            ),
        )

    return _tratar(executar)


# ---------------------------------------------------------------------------
# API — Materiais compósitos
# ---------------------------------------------------------------------------

@app.route("/api/material/laminado", methods=["POST"])
def api_material_laminado():
    def executar():
        dados = request.get_json(force=True)
        if MATERIAIS is None or Lamina is None or LaminaOrtotropica is None:
            raise ErroExpressao("módulo de materiais compósitos indisponível neste ambiente.")

        def escalar(nome, padrao):
            """Campo numérico simples (aceita 'G' como multiplicador de 1e9)."""
            texto = str(dados.get(nome, "") or "").strip()
            if not texto:
                return padrao
            fator = 1.0
            if texto.upper().endswith("G"):
                fator, texto = 1e9, texto[:-1].strip()
            return numero(texto, nome) * fator

        material = LaminaOrtotropica(
            e1=escalar("e1", 140e9),
            e2=escalar("e2", 10e9),
            g12=escalar("g12", 7e9),
            nu12=numero(dados.get("nu12", 0.3), "ν12"),
            alpha1=numero(dados.get("alpha1", 0.0), "α1"),
            alpha2=numero(dados.get("alpha2", 22.5e-6), "α2"),
            x_t=escalar("xt", 1500e6),
            x_c=escalar("xc", 1200e6),
            y_t=escalar("yt", 50e6),
            y_c=escalar("yc", 200e6),
            s=escalar("s", 70e6),
        )
        angulos = vetor(dados.get("angulos", "[0, 90, 0]"), "ângulos das camadas")
        espessura = numero(dados.get("espessura", 0.15e-3), "espessura por camada")
        pilhas = [Lamina(material=material, angulo_graus=float(angulo), espessura=espessura)
                  for angulo in angulos]
        nx = escalar("nx", 1e6)
        resultado = MATERIAIS.analisar_laminado(pilhas, resultantes_membrana=(nx, 0.0, 0.0))

        resumo = {}
        for chave, valor in resultado.items():
            if isinstance(valor, np.ndarray):
                resumo[chave] = np.round(np.asarray(valor, dtype=float), 4).tolist()
            elif isinstance(valor, (int, float)):
                resumo[chave] = formatar_numero(valor)
            elif isinstance(valor, list) and valor and isinstance(valor[0], dict):
                # resultados por camada (topo/base) — reduzidos a escalares arredondados
                camadas = []
                for item in valor:
                    camada = {}
                    for lado, dados_lado in item.items():
                        if isinstance(dados_lado, dict):
                            camada[lado] = {
                                kk: (np.round(np.asarray(vv, dtype=float), 4).tolist()
                                     if isinstance(vv, np.ndarray) else float(vv))
                                for kk, vv in dados_lado.items() if kk != "z"
                            }
                    camadas.append(camada)
                resumo[chave] = camadas
            elif isinstance(valor, list):
                continue
            else:
                try:
                    json_serializar(valor)
                    resumo[chave] = valor
                except Exception:
                    continue
        return _resposta_ok(
            resumo=resumo,
            detalhe=(f"Laminado [{', '.join(formatar_numero(a) for a in angulos)}]°, "
                     f"{len(pilhas)} camada(s) de {formatar_numero(espessura * 1e3)} mm, "
                     f"Nx = {formatar_numero(nx / 1e6)} MN/m"),
        )

    return _tratar(executar)


def json_serializar(valor):
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, (np.floating, np.integer)):
        return valor.item()
    if isinstance(valor, dict):
        return {str(k): json_serializar(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [json_serializar(v) for v in valor]
    return valor


# ---------------------------------------------------------------------------
# API — Álgebra linear (requer TensorFlow)
# ---------------------------------------------------------------------------

@app.route("/api/matriz/determinante", methods=["POST"])
def api_matriz_determinante():
    def executar():
        if MATRIZES is None:
            raise ErroExpressao(
                "módulo de matrizes indisponível: instale TensorFlow (requirements.txt)."
            )
        A = matriz(request.get_json(force=True).get("matriz", ""), "matriz A")
        det = MATRIZES.Determinante(A)
        return _resposta_ok(det=formatar_numero(det))

    return _tratar(executar)


@app.route("/api/matriz/inversa", methods=["POST"])
def api_matriz_inversa():
    def executar():
        if MATRIZES is None:
            raise ErroExpressao(
                "módulo de matrizes indisponível: instale TensorFlow (requirements.txt)."
            )
        A = matriz(request.get_json(force=True).get("matriz", ""), "matriz A")
        inversa = MATRIZES.Inversa(A)
        inversa = np.asarray(inversa, dtype=float)
        linhas = [[formatar_numero(v) for v in linha] for linha in np.atleast_2d(inversa)]
        return _resposta_ok(inversa=linhas)

    return _tratar(executar)


@app.route("/api/matriz/sistema", methods=["POST"])
def api_matriz_sistema():
    def executar():
        if MATRIZES is None:
            raise ErroExpressao(
                "módulo de matrizes indisponível: instale TensorFlow (requirements.txt)."
            )
        dados = request.get_json(force=True)
        A = matriz(dados.get("matriz", ""), "matriz A")
        b = vetor(dados.get("b", ""), "vetor b")
        x = MATRIZES.GaussJordan(A, b)
        x = np.asarray(x, dtype=float).ravel()
        return _resposta_ok(
            solucao=[formatar_numero(v) for v in x],
            residual=formatar_numero(float(np.linalg.norm(np.asarray(A) @ x - np.asarray(b)))),
        )

    return _tratar(executar)


# ---------------------------------------------------------------------------
# API — Constantes
# ---------------------------------------------------------------------------

@app.route("/api/constantes")
def api_constantes():
    if CONST is None:
        return _resposta_erro("módulo de constantes indisponível neste ambiente.", status=500)
    itens = []
    for nome, valor in vars(CONST).items():
        if nome.startswith("_"):
            continue
        itens.append({"nome": nome, "valor": formatar_numero(valor)})
    itens.sort(key=lambda item: item["nome"])
    return _resposta_ok(constantes=itens)


# ---------------------------------------------------------------------------
# API — Tabela periódica & fluidos
# ---------------------------------------------------------------------------

@app.route("/api/elementos/buscar")
def api_elementos_buscar():
    if ElementosMateriais is None:
        return _resposta_erro("módulo de elementos indisponível neste ambiente.", status=500)
    termo = (request.args.get("q") or "").strip()
    try:
        tabela = ElementosMateriais.carregar_elemento()
    except Exception as exc:
        return _resposta_erro(f"falha ao carregar dados locais: {exc}", status=500)
    resultados = []
    for simbolo, dados in tabela.items():
        if not isinstance(dados, dict):
            continue
        nome = str(dados.get("name", dados.get("nome", simbolo)))
        if termo.lower() in simbolo.lower() or termo.lower() in nome.lower():
            resultados.append({
                "simbolo": simbolo,
                "nome": nome,
                "numero_atomico": dados.get("atomic_number", dados.get("number")),
                "massa": dados.get("atomic_weight", dados.get("mass")),
                "categoria": dados.get("category", dados.get("type", "")),
            })
        if len(resultados) >= 30:
            break
    return _resposta_ok(resultados=resultados, total=len(tabela))


@app.route("/api/elementos/detalhe/<simbolo>")
def api_elemento_detalhe(simbolo):
    if ElementosMateriais is None:
        return _resposta_erro("módulo de elementos indisponível neste ambiente.", status=500)
    try:
        dados = ElementosMateriais.carregar_elemento(simbolo.upper() if simbolo.islower() else simbolo)
    except Exception:
        try:
            dados = ElementosMateriais.carregar_elemento(simbolo)
        except Exception as exc:
            return _resposta_erro(f"elemento não encontrado: {exc}")
    return _resposta_ok(simbolo=simbolo, dados=json_serializar(dados))


@app.route("/api/fluidos/lista")
def api_fluidos_lista():
    if ElementosMateriais is None:
        return _resposta_erro("módulo de fluidos indisponível neste ambiente.", status=500)
    try:
        dados = ElementosMateriais.carregar_fluido()
    except Exception as exc:
        return _resposta_erro(f"falha ao carregar fluidos: {exc}", status=500)
    nomes = []
    for chave, valor in dados.items():
        if isinstance(valor, dict):
            nomes.append(chave)
    return _resposta_ok(fluidos=nomes)


@app.route("/api/fluidos/detalhe/<nome>")
def api_fluido_detalhe(nome):
    if ElementosMateriais is None:
        return _resposta_erro("módulo de fluidos indisponível neste ambiente.", status=500)
    try:
        dados = ElementosMateriais.carregar_fluido(nome)
    except Exception as exc:
        return _resposta_erro(f"fluido não encontrado: {exc}")
    return _resposta_ok(dados=json_serializar(dados))


# ---------------------------------------------------------------------------
# API — Escoamento Venturi 3D
# ---------------------------------------------------------------------------

@app.route("/api/venturi", methods=["POST"])
def api_venturi():
    def executar():
        dados = request.get_json(force=True)
        if SimulacaoTempoRealVenturi3D is None:
            raise ErroExpressao("simulador Venturi indisponível neste ambiente.")
        simulador = SimulacaoTempoRealVenturi3D(
            p0=numero(dados.get("p0", 500000), "pressão de estagnação p₀"),
            T0=numero(dados.get("T0", 900), "temperatura de estagnação T₀"),
            gamma=numero(dados.get("gamma", 1.4), "razão γ"),
            L_total=numero(dados.get("L", 2.0), "comprimento L"),
            A_entrada=numero(dados.get("A_in", 0.1), "área de entrada"),
            A_garganta=numero(dados.get("A_garg", 0.02), "área de garganta"),
        )
        n = 200
        xs = np.linspace(0, simulador.L_total, n)
        mach, pressao, temperatura, densidade = [], [], [], []
        for x in xs:
            estado = simulador.obter_estado_fluido(float(x), 0.0, 0.0)
            mach.append(estado["Mach"])
            pressao.append(estado["Pressao"])
            temperatura.append(estado["Temperatura"])
            densidade.append(estado["Densidade"])
        perfil_area = [simulador.area_transversal(float(x)) for x in xs]
        raio_max = max(math.sqrt(a / math.pi) for a in perfil_area)
        contorno_superior = [math.sqrt(a / math.pi) for a in perfil_area]
        contorno_inferior = [-math.sqrt(a / math.pi) for a in perfil_area]
        return _resposta_ok(
            regime=getattr(simulador, "regime", None) or "calculado",
            saida={
                "Mach_saida": formatar_numero(mach[-1]),
                "Mach_garganta": formatar_numero(mach[n // 2]),
                "Pressao_saida_MPa": formatar_numero(pressao[-1] / 1e6),
                "Temperatura_saida_K": formatar_numero(temperatura[-1]),
            },
            eixo_x=_numpy_para_lista(xs),
            perfis={
                "mach": _numpy_para_lista(mach),
                "pressao_kpa": _numpy_para_lista([p / 1e3 for p in pressao]),
                "temperatura": _numpy_para_lista(temperatura),
                "densidade": _numpy_para_lista(densidade),
            },
            geometria={
                "x": _numpy_para_lista(xs),
                "superior": _numpy_para_lista(contorno_superior),
                "inferior": _numpy_para_lista(contorno_inferior),
                "escala": raio_max,
            },
        )

    return _tratar(executar)


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------

def criar_app():
    return app


if __name__ == "__main__":
    porta = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"Ω OMEGA Utils — painel em http://127.0.0.1:{porta}")
    app.run(host="127.0.0.1", port=porta, debug=False)
