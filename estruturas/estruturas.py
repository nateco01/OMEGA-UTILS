"""Modelos parametrizados de estabilidade, fadiga e crescimento de trinca.

Cada rotina corresponde a uma teoria especifica e declara suas hipoteses.
Os resultados sao calculos analiticos preliminares, nao substituem normas,
curvas de ensaio, elementos finitos ou validacao experimental.
"""

import math
from numbers import Real
from typing import Sequence

import numpy as np


def _positivo(valor: float, nome: str, permitir_zero: bool = False) -> float:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"{nome} deve ser numerico.") from None
    if not math.isfinite(numero) or numero < 0.0 or (numero == 0.0 and not permitir_zero):
        operador = "nao negativo" if permitir_zero else "positivo"
        raise ValueError(f"{nome} deve ser finito e {operador}.")
    return numero


class AnaliseEstruturalSolver:
    """Formulas independentes de dados de material e geometria predefinidos."""

    @staticmethod
    def carga_critica_euler(
        modulo_elasticidade: float,
        momento_inercia: float,
        comprimento: float,
        fator_comprimento_efetivo: float = 1.0,
    ) -> float:
        """Pcr de coluna esbelta elastica: pi^2 E I / (K L)^2, em N."""
        e = _positivo(modulo_elasticidade, "modulo_elasticidade")
        inercia = _positivo(momento_inercia, "momento_inercia")
        comprimento = _positivo(comprimento, "comprimento")
        k = _positivo(fator_comprimento_efetivo, "fator_comprimento_efetivo")
        return math.pi**2 * e * inercia / (k * comprimento) ** 2

    @staticmethod
    def flambagem_placa_isotropica(
        modulo_elasticidade: float,
        poisson: float,
        espessura: float,
        largura: float,
        coeficiente_k: float,
    ) -> dict[str, float]:
        """Placa fina isotropica sob compressao uniaxial, Ncr=k*pi^2*D/b^2.

        O coeficiente k deve ser obtido para a razao de aspecto e apoios reais.
        Retorna rigidez D (N m), resultante critica Nx (N/m) e tensao (Pa).
        """
        e = _positivo(modulo_elasticidade, "modulo_elasticidade")
        t = _positivo(espessura, "espessura")
        b = _positivo(largura, "largura")
        k = _positivo(coeficiente_k, "coeficiente_k")
        nu = float(poisson)
        if not math.isfinite(nu) or not -1.0 < nu < 0.5:
            raise ValueError("poisson deve satisfazer -1 < nu < 0.5.")
        d = e * t**3 / (12.0 * (1.0 - nu**2))
        n_critico = k * math.pi**2 * d / b**2
        return {
            "rigidez_flexional": d,
            "resultante_critica": n_critico,
            "tensao_critica": n_critico / t,
        }

    @staticmethod
    def flambagem_placa_ortotropica_simplesmente_apoiada(
        matriz_d: Sequence[Sequence[float]],
        comprimento: float,
        largura: float,
        modos_comprimento: int = 4,
        modos_largura: int = 4,
        espessura: float | None = None,
    ) -> dict[str, float]:
        """Nx critico por Ritz multimodal para placa CLT simplesmente apoiada.

        matriz_d e a submatriz D 3x3 de CLT em N m; Nx,crit fica em N/m.
        A base usa senos duplos e captura os acoplamentos D16/D26. Aumentar os
        limites modais melhora a aproximacao; o valor de Ritz converge por
        cima para o menor autovalor do problema idealizado.
        """
        d = np.asarray(matriz_d, dtype=float)
        if d.shape != (3, 3) or not np.all(np.isfinite(d)):
            raise ValueError("matriz_d deve ser uma matriz 3x3 finita.")
        a = _positivo(comprimento, "comprimento")
        b = _positivo(largura, "largura")
        if any(
            isinstance(modos, bool) or not isinstance(modos, int) or modos < 1
            for modos in (modos_comprimento, modos_largura)
        ):
            raise ValueError("Os limites modais devem ser inteiros positivos.")
        if not np.allclose(d, d.T, rtol=1e-9, atol=1e-12):
            raise ValueError("matriz_d deve ser simetrica.")
        try:
            np.linalg.cholesky(d)
        except np.linalg.LinAlgError as erro:
            raise ValueError("matriz_d deve ser positiva definida.") from erro
        if espessura is not None:
            t = _positivo(espessura, "espessura")
        else:
            t = None

        pontos = max(24, 6 * max(modos_comprimento, modos_largura))
        nos, pesos = np.polynomial.legendre.leggauss(pontos)
        xs = 0.5 * a * (nos + 1.0)
        ys = 0.5 * b * (nos + 1.0)
        wx = 0.5 * a * pesos
        wy = 0.5 * b * pesos
        x_mesh, y_mesh = np.meshgrid(xs, ys, indexing="ij")
        peso_area = np.outer(wx, wy).ravel()
        x_points, y_points = x_mesh.ravel(), y_mesh.ravel()

        pares_modos = [
            (m, n)
            for m in range(1, modos_comprimento + 1)
            for n in range(1, modos_largura + 1)
        ]
        curvaturas = []
        inclinacoes_x = []
        for m, n in pares_modos:
            alpha = m * math.pi / a
            beta = n * math.pi / b
            sin_x, cos_x = np.sin(alpha * x_points), np.cos(alpha * x_points)
            sin_y, cos_y = np.sin(beta * y_points), np.cos(beta * y_points)
            curvaturas.append(
                np.column_stack(
                    (
                        -alpha**2 * sin_x * sin_y,
                        -beta**2 * sin_x * sin_y,
                        2.0 * alpha * beta * cos_x * cos_y,
                    )
                )
            )
            inclinacoes_x.append(alpha * cos_x * sin_y)

        curvaturas = np.asarray(curvaturas)
        inclinacoes_x = np.asarray(inclinacoes_x)
        rigidez_modal = np.einsum(
            "ipk,kl,jpl,p->ij", curvaturas, d, curvaturas, peso_area
        )
        rigidez_membrana = np.einsum(
            "ip,jp,p->ij", inclinacoes_x, inclinacoes_x, peso_area
        )
        try:
            chol = np.linalg.cholesky(rigidez_membrana)
            chol_inv = np.linalg.solve(chol, np.eye(chol.shape[0]))
            problema_simetrico = chol_inv @ rigidez_modal @ chol_inv.T
            autovalores, autovetores = np.linalg.eigh(problema_simetrico)
        except np.linalg.LinAlgError as erro:
            raise ValueError("Falha ao montar o problema generalizado de flambagem.") from erro

        validos = np.flatnonzero(np.isfinite(autovalores) & (autovalores > 0.0))
        if validos.size == 0:
            raise ValueError("Nenhuma carga critica positiva foi encontrada.")
        indice = int(validos[0])
        n_critico = float(autovalores[indice])
        coeficientes = chol_inv.T @ autovetores[:, indice]
        modo_dominante = pares_modos[int(np.argmax(np.abs(coeficientes)))]
        resultado = {
            "resultante_critica": n_critico,
            "modo_dominante_comprimento": float(modo_dominante[0]),
            "modo_dominante_largura": float(modo_dominante[1]),
        }
        if t is not None:
            resultado["tensao_critica_media"] = n_critico / t
        return resultado

    @staticmethod
    def amplitude_goodman(
        amplitude_tensao: float,
        tensao_media: float,
        resistencia_tracao: float,
    ) -> float:
        """Amplitude equivalente totalmente alternada pelo criterio de Goodman.

        Requer tensao media abaixo da resistencia ultima a tracao; compressao
        media e permitida, mas o criterio nao modela fechamento de trinca.
        """
        amplitude = _positivo(amplitude_tensao, "amplitude_tensao", permitir_zero=True)
        sut = _positivo(resistencia_tracao, "resistencia_tracao")
        media = float(tensao_media)
        if not math.isfinite(media):
            raise ValueError("tensao_media deve ser finita.")
        divisor = 1.0 - media / sut
        if divisor <= 0.0:
            raise ValueError("Goodman nao esta definido para tensao media >= Sut.")
        return amplitude / divisor

    @staticmethod
    def vida_fadiga_basquin(
        amplitude_tensao: float,
        coeficiente_fadiga: float,
        expoente_basquin: float,
    ) -> float:
        """Vida Nf por sigma_a=sigma_f'*(2Nf)^b; retorna ciclos completos."""
        amplitude = _positivo(amplitude_tensao, "amplitude_tensao", permitir_zero=True)
        coeficiente = _positivo(coeficiente_fadiga, "coeficiente_fadiga")
        expoente = float(expoente_basquin)
        if not math.isfinite(expoente) or expoente >= 0.0:
            raise ValueError("expoente_basquin deve ser negativo e finito.")
        if amplitude == 0.0:
            return math.inf
        return 0.5 * (amplitude / coeficiente) ** (1.0 / expoente)

    @staticmethod
    def dano_acumulado_miner(
        ciclos_aplicados: Sequence[float], vidas_fadiga: Sequence[float]
    ) -> dict[str, float | bool]:
        """Regra linear de Miner D=sum(ni/Ni), para ciclos ja identificados."""
        ciclos = np.asarray(ciclos_aplicados, dtype=float)
        vidas = np.asarray(vidas_fadiga, dtype=float)
        if ciclos.ndim != 1 or vidas.ndim != 1 or ciclos.shape != vidas.shape or ciclos.size == 0:
            raise ValueError("ciclos_aplicados e vidas_fadiga devem ser vetores nao vazios iguais.")
        if not np.all(np.isfinite(ciclos)) or np.any(ciclos < 0.0):
            raise ValueError("Os ciclos aplicados devem ser finitos e nao negativos.")
        if np.any(np.isnan(vidas)) or np.any(vidas <= 0.0):
            raise ValueError("As vidas de fadiga devem ser positivas (infinito e permitido).")
        dano = float(np.sum(ciclos / vidas))
        return {"dano": dano, "falha_prevista": dano >= 1.0}

    @staticmethod
    def fator_intensidade_tensao_modo_i(
        tensao_tracao: float, comprimento_trinca: float, fator_geometrico: float = 1.0
    ) -> float:
        """K_I=Y*sigma*sqrt(pi*a), em Pa sqrt(m), para tracao Mode I."""
        sigma = _positivo(tensao_tracao, "tensao_tracao", permitir_zero=True)
        a = _positivo(comprimento_trinca, "comprimento_trinca")
        y = _positivo(fator_geometrico, "fator_geometrico")
        return y * sigma * math.sqrt(math.pi * a)

    @classmethod
    def comprimento_critico_trinca(
        cls, tenacidade_fratura: float, tensao_maxima: float, fator_geometrico: float = 1.0
    ) -> float:
        """Tamanho a para K_I=K_IC; LEFM, geometria Y constante e Mode I."""
        kic = _positivo(tenacidade_fratura, "tenacidade_fratura")
        sigma = _positivo(tensao_maxima, "tensao_maxima")
        y = _positivo(fator_geometrico, "fator_geometrico")
        return (kic / (y * sigma)) ** 2 / math.pi

    @classmethod
    def crescimento_trinca_paris(
        cls,
        comprimento_inicial: float,
        comprimento_critico: float,
        delta_tensao: float,
        coeficiente_paris: float,
        expoente_paris: float,
        ciclos_aplicados: float,
        fator_geometrico: float = 1.0,
    ) -> dict[str, float | bool]:
        """Crescimento analitico sob amplitude constante, regime Paris e Y constante.

        Convencao: da/dN=C*(DeltaK)^m com DeltaK em Pa sqrt(m); C deve usar
        unidades consistentes. Nao inclui limiar DeltaK_th nem regime de fratura.
        """
        a0 = _positivo(comprimento_inicial, "comprimento_inicial")
        ac = _positivo(comprimento_critico, "comprimento_critico")
        dsigma = _positivo(delta_tensao, "delta_tensao", permitir_zero=True)
        c_paris = _positivo(coeficiente_paris, "coeficiente_paris", permitir_zero=True)
        m = _positivo(expoente_paris, "expoente_paris")
        n_aplicado = _positivo(ciclos_aplicados, "ciclos_aplicados", permitir_zero=True)
        y = _positivo(fator_geometrico, "fator_geometrico")
        if ac <= a0:
            raise ValueError("comprimento_critico deve exceder comprimento_inicial.")
        taxa = c_paris * (y * dsigma * math.sqrt(math.pi)) ** m
        if taxa == 0.0:
            return {
                "comprimento_final": a0,
                "ciclos_ate_critico": math.inf,
                "atingiu_critico": False,
            }
        expoente = 1.0 - 0.5 * m
        if math.isclose(expoente, 0.0, abs_tol=1e-12):
            ciclos_ate_critico = math.log(ac / a0) / taxa
            a_final = a0 * math.exp(taxa * n_aplicado)
        else:
            ciclos_ate_critico = (ac**expoente - a0**expoente) / (expoente * taxa)
            termo = a0**expoente + expoente * taxa * n_aplicado
            a_final = math.inf if termo <= 0.0 else termo ** (1.0 / expoente)
        atingiu = n_aplicado >= ciclos_ate_critico or a_final >= ac
        if atingiu:
            a_final = ac
        return {
            "comprimento_final": float(a_final),
            "ciclos_ate_critico": float(ciclos_ate_critico),
            "atingiu_critico": bool(atingiu),
        }

    @staticmethod
    def dilatacao_termica(
        comprimento: float, coeficiente_expansao: float, delta_temperatura: float
    ) -> float:
        """Delta L=L*alpha*DeltaT para expansao linear uniforme isotropica."""
        l = _positivo(comprimento, "comprimento", permitir_zero=True)
        alpha = float(coeficiente_expansao)
        delta_t = float(delta_temperatura)
        if not math.isfinite(alpha) or not math.isfinite(delta_t):
            raise ValueError("coeficiente_expansao e delta_temperatura devem ser finitos.")
        return l * alpha * delta_t

    @staticmethod
    def tensao_termica_uniaxial_restrita(
        modulo_elasticidade: float,
        coeficiente_expansao: float,
        delta_temperatura: float,
        fracao_restricao: float = 1.0,
    ) -> float:
        """sigma=-r*E*alpha*DeltaT; r=1 totalmente preso, r=0 livre."""
        e = _positivo(modulo_elasticidade, "modulo_elasticidade")
        alpha = float(coeficiente_expansao)
        delta_t = float(delta_temperatura)
        restricao = float(fracao_restricao)
        if not all(math.isfinite(v) for v in (alpha, delta_t, restricao)):
            raise ValueError("Os parametros termicos devem ser finitos.")
        if not 0.0 <= restricao <= 1.0:
            raise ValueError("fracao_restricao deve estar entre 0 e 1.")
        return -restricao * e * alpha * delta_t