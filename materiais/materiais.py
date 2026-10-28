"""Modelos constitutivos de laminados ortotropicos em estado plano de tensao.

As rotinas implementam teoria classica de laminados (CLT): laminas lineares,
perfeitamente aderidas, pequenas deformacoes e deformacao normal constante
atraves de cada lamina. Nao modelam tensoes interlaminares, dano progressivo,
delaminacao ou efeitos viscoelasticos.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class LaminaOrtotropica:
    """Propriedades em eixos de fibra e matriz, em unidades SI."""

    e1: float
    e2: float
    g12: float
    nu12: float
    alpha1: float
    alpha2: float
    x_t: float
    x_c: float
    y_t: float
    y_c: float
    s: float

    def __post_init__(self):
        valores = (
            self.e1,
            self.e2,
            self.g12,
            self.nu12,
            self.alpha1,
            self.alpha2,
            self.x_t,
            self.x_c,
            self.y_t,
            self.y_c,
            self.s,
        )
        if not all(math.isfinite(float(valor)) for valor in valores):
            raise ValueError("As propriedades da lamina devem ser finitas.")
        if min(self.e1, self.e2, self.g12, self.x_t, self.x_c, self.y_t, self.y_c, self.s) <= 0:
            raise ValueError("Modulos e resistencias da lamina devem ser positivos.")
        nu21 = self.nu12 * self.e2 / self.e1
        if 1.0 - self.nu12 * nu21 <= 0.0:
            raise ValueError("As constantes elasticas violam a estabilidade ortotropica.")


@dataclass(frozen=True)
class Lamina:
    """Uma camada do laminado; angulo positivo gira o eixo 1 a partir de x."""

    material: LaminaOrtotropica
    angulo_graus: float
    espessura: float

    def __post_init__(self):
        if not math.isfinite(float(self.angulo_graus)):
            raise ValueError("O angulo da lamina deve ser finito.")
        if not math.isfinite(float(self.espessura)) or self.espessura <= 0.0:
            raise ValueError("A espessura de cada lamina deve ser positiva e finita.")


class MateriaisCompositosSolver:
    """Teoria classica de laminados e criterios usuais de resistencia."""

    @staticmethod
    def matriz_q(material: LaminaOrtotropica) -> np.ndarray:
        """Matriz reduzida Q da lamina ortotropica em estado plano de tensao."""
        nu21 = material.nu12 * material.e2 / material.e1
        divisor = 1.0 - material.nu12 * nu21
        return np.array(
            [
                [material.e1 / divisor, material.nu12 * material.e2 / divisor, 0.0],
                [material.nu12 * material.e2 / divisor, material.e2 / divisor, 0.0],
                [0.0, 0.0, material.g12],
            ],
            dtype=float,
        )

    @classmethod
    def matriz_q_transformada(
        cls, material: LaminaOrtotropica, angulo_graus: float
    ) -> np.ndarray:
        """Transforma Q para eixos globais; valido para qualquer orientacao."""
        q11, q22, q12, q66 = (
            cls.matriz_q(material)[0, 0],
            cls.matriz_q(material)[1, 1],
            cls.matriz_q(material)[0, 1],
            cls.matriz_q(material)[2, 2],
        )
        theta = math.radians(angulo_graus)
        m, n = math.cos(theta), math.sin(theta)
        m2, n2 = m * m, n * n
        m4, n4 = m2 * m2, n2 * n2
        mn = m * n
        qbar = np.empty((3, 3), dtype=float)
        qbar[0, 0] = q11 * m4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * n4
        qbar[1, 1] = q11 * n4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * m4
        qbar[0, 1] = qbar[1, 0] = (
            (q11 + q22 - 4.0 * q66) * m2 * n2 + q12 * (m4 + n4)
        )
        qbar[0, 2] = qbar[2, 0] = (
            (q11 - q12 - 2.0 * q66) * m**3 * n
            - (q22 - q12 - 2.0 * q66) * m * n**3
        )
        qbar[1, 2] = qbar[2, 1] = (
            (q11 - q12 - 2.0 * q66) * m * n**3
            - (q22 - q12 - 2.0 * q66) * m**3 * n
        )
        qbar[2, 2] = (
            (q11 + q22 - 2.0 * q12 - 2.0 * q66) * m2 * n2
            + q66 * (m4 + n4)
        )
        return qbar

    @staticmethod
    def _alpha_global(material: LaminaOrtotropica, angulo_graus: float) -> np.ndarray:
        theta = math.radians(angulo_graus)
        m, n = math.cos(theta), math.sin(theta)
        return np.array(
            [
                material.alpha1 * m * m + material.alpha2 * n * n,
                material.alpha1 * n * n + material.alpha2 * m * m,
                2.0 * m * n * (material.alpha1 - material.alpha2),
            ],
            dtype=float,
        )

    @staticmethod
    def transformar_tensao_local(
        tensao_global: Sequence[float], angulo_graus: float
    ) -> np.ndarray:
        """Transforma [sx, sy, txy] para [s1, s2, t12] da lamina."""
        tensao = np.asarray(tensao_global, dtype=float)
        if tensao.shape != (3,) or not np.all(np.isfinite(tensao)):
            raise ValueError("tensao_global deve conter tres valores finitos.")
        theta = math.radians(angulo_graus)
        m, n = math.cos(theta), math.sin(theta)
        sx, sy, txy = tensao
        return np.array(
            [
                m * m * sx + n * n * sy + 2.0 * m * n * txy,
                n * n * sx + m * m * sy - 2.0 * m * n * txy,
                -m * n * sx + m * n * sy + (m * m - n * n) * txy,
            ]
        )

    @staticmethod
    def indice_tensao_maxima(
        tensao_local: Sequence[float], material: LaminaOrtotropica
    ) -> dict[str, object]:
        """Utilizacao por tensao maxima; falha nominal quando utilizacao >= 1."""
        tensao = np.asarray(tensao_local, dtype=float)
        if tensao.shape != (3,) or not np.all(np.isfinite(tensao)):
            raise ValueError("tensao_local deve conter tres valores finitos.")
        limites = np.array(
            [
                material.x_t if tensao[0] >= 0 else material.x_c,
                material.y_t if tensao[1] >= 0 else material.y_c,
                material.s,
            ]
        )
        utilizacoes = np.abs(tensao) / limites
        return {
            "utilizacao": float(np.max(utilizacoes)),
            "componentes": utilizacoes,
            "falha_prevista": bool(np.max(utilizacoes) >= 1.0),
        }

    @staticmethod
    def indice_tsai_hill(
        tensao_local: Sequence[float], material: LaminaOrtotropica
    ) -> float:
        """Indice Tsai-Hill plane-stress com resistencia escolhida pelo sinal."""
        tensao = np.asarray(tensao_local, dtype=float)
        if tensao.shape != (3,) or not np.all(np.isfinite(tensao)):
            raise ValueError("tensao_local deve conter tres valores finitos.")
        s1, s2, t12 = tensao
        x = material.x_t if s1 >= 0 else material.x_c
        y = material.y_t if s2 >= 0 else material.y_c
        return float((s1 / x) ** 2 - s1 * s2 / x**2 + (s2 / y) ** 2 + (t12 / material.s) ** 2)

    @staticmethod
    def indice_tsai_wu(
        tensao_local: Sequence[float],
        material: LaminaOrtotropica,
        f12: float | None = None,
    ) -> float:
        """Indice Tsai-Wu; f12 default e a aproximacao usual, nao dado universal."""
        tensao = np.asarray(tensao_local, dtype=float)
        if tensao.shape != (3,) or not np.all(np.isfinite(tensao)):
            raise ValueError("tensao_local deve conter tres valores finitos.")
        f1 = 1.0 / material.x_t - 1.0 / material.x_c
        f2 = 1.0 / material.y_t - 1.0 / material.y_c
        f11 = 1.0 / (material.x_t * material.x_c)
        f22 = 1.0 / (material.y_t * material.y_c)
        f66 = 1.0 / material.s**2
        if f12 is None:
            f12 = -0.5 * math.sqrt(f11 * f22)
        if not math.isfinite(f12):
            raise ValueError("f12 deve ser finito.")
        s1, s2, t12 = tensao
        return float(
            f1 * s1
            + f2 * s2
            + f11 * s1**2
            + f22 * s2**2
            + f66 * t12**2
            + 2.0 * f12 * s1 * s2
        )

    @classmethod
    def analisar_laminado(
        cls,
        pilhas: Sequence[Lamina],
        resultantes_membrana: Sequence[float] = (0.0, 0.0, 0.0),
        momentos: Sequence[float] = (0.0, 0.0, 0.0),
        delta_temperatura: float | Callable[[float], float] = 0.0,
    ) -> dict[str, object]:
        """Resolve CLT [A B; B D] com cargas e temperatura atraves da espessura.

        resultantes_membrana: [Nx, Ny, Nxy] em N/m.
        momentos: [Mx, My, Mxy] em N.
        delta_temperatura: valor uniforme em K ou funcao deltaT(z) em K.
        """
        if not pilhas or not all(isinstance(pilha, Lamina) for pilha in pilhas):
            raise ValueError("pilhas deve conter ao menos um objeto Lamina.")
        carga_n = np.asarray(resultantes_membrana, dtype=float)
        carga_m = np.asarray(momentos, dtype=float)
        if carga_n.shape != (3,) or carga_m.shape != (3,):
            raise ValueError("resultantes_membrana e momentos devem ter tres componentes.")
        if not np.all(np.isfinite(carga_n)) or not np.all(np.isfinite(carga_m)):
            raise ValueError("As cargas devem ser finitas.")
        if not callable(delta_temperatura) and (
            not isinstance(delta_temperatura, Real)
            or not math.isfinite(float(delta_temperatura))
        ):
            raise ValueError("delta_temperatura deve ser finito ou uma funcao deltaT(z).")

        espessura_total = sum(pilha.espessura for pilha in pilhas)
        z_limites = [-0.5 * espessura_total]
        for pilha in pilhas:
            z_limites.append(z_limites[-1] + pilha.espessura)

        a = np.zeros((3, 3))
        b = np.zeros((3, 3))
        d = np.zeros((3, 3))
        n_termico = np.zeros(3)
        m_termico = np.zeros(3)
        qbar_pilhas = []
        alpha_pilhas = []
        nos_gauss, pesos_gauss = np.polynomial.legendre.leggauss(3)

        for indice, pilha in enumerate(pilhas):
            z0, z1 = z_limites[indice : indice + 2]
            qbar = cls.matriz_q_transformada(pilha.material, pilha.angulo_graus)
            alpha = cls._alpha_global(pilha.material, pilha.angulo_graus)
            a += qbar * (z1 - z0)
            b += 0.5 * qbar * (z1**2 - z0**2)
            d += (qbar / 3.0) * (z1**3 - z0**3)
            qbar_pilhas.append(qbar)
            alpha_pilhas.append(alpha)

            for no, peso in zip(nos_gauss, pesos_gauss):
                z = 0.5 * (z1 + z0) + 0.5 * (z1 - z0) * no
                delta_t = (
                    float(delta_temperatura(z))
                    if callable(delta_temperatura)
                    else float(delta_temperatura)
                )
                if not math.isfinite(delta_t):
                    raise ValueError("delta_temperatura(z) deve retornar valores finitos.")
                fator = 0.5 * (z1 - z0) * peso * delta_t
                n_termico += qbar @ alpha * fator
                m_termico += z * (qbar @ alpha) * fator

        abd = np.block([[a, b], [b, d]])
        rhs = np.concatenate((carga_n + n_termico, carga_m + m_termico))
        try:
            resposta = np.linalg.solve(abd, rhs)
        except np.linalg.LinAlgError as erro:
            raise ValueError("A matriz ABD e singular; verifique a pilha e propriedades.") from erro

        deformacao_meio = resposta[:3]
        curvatura = resposta[3:]
        resultados_pilhas = []
        for indice, pilha in enumerate(pilhas):
            z0, z1 = z_limites[indice : indice + 2]
            qbar = qbar_pilhas[indice]
            alpha = alpha_pilhas[indice]
            lados = {}
            for nome, z in (("inferior", z0), ("superior", z1)):
                deformacao_global = deformacao_meio + z * curvatura
                delta_t = (
                    float(delta_temperatura(z))
                    if callable(delta_temperatura)
                    else float(delta_temperatura)
                )
                tensao_global = qbar @ (deformacao_global - alpha * delta_t)
                tensao_local = cls.transformar_tensao_local(
                    tensao_global, pilha.angulo_graus
                )
                lados[nome] = {
                    "z": float(z),
                    "deformacao_global": deformacao_global,
                    "tensao_global": tensao_global,
                    "tensao_local": tensao_local,
                }
            resultados_pilhas.append(lados)

        return {
            "A": a,
            "B": b,
            "D": d,
            "ABD": abd,
            "resultantes_termicas": np.concatenate((n_termico, m_termico)),
            "deformacao_meio": deformacao_meio,
            "curvatura": curvatura,
            "pilhas": resultados_pilhas,
            "limitacoes": [
                "CLT linear em estado plano de tensao, pequenas deformacoes e aderencia perfeita.",
                "Nao calcula tensoes interlaminares, delaminacao nem dano progressivo.",
                "Tsai-Wu requer F12 experimental; o valor default e apenas uma estimativa comum.",
            ],
        }