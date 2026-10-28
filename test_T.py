"""Triagem 3D de tensoes para um casco cilindrico carbono/niquel.

Este e um modelo analitico de membrana, nao um solver FEM. O escoamento usa
Cp de cilindro infinito inviscido/incompressivel; a estrutura usa laminado
linear elastico, pequenas deformacoes e teoria de casca fina. Geometria,
propriedades e resistencia sao valores ilustrativos editaveis, nao dados de
fabricacao. A estimativa nao verifica flambagem, juntas, dano, fadiga,
impacto, transferencia termica nem efeitos viscosos/compressibilidade.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from constantes.constantes import Constantes
from derivadas.derivadas import DerivadasSolver
from equacoes.equacoes import EDOsSolver, EquacoesSolver
from integrais.integrais import IntegralSolverDefinidas, IntegralSolverIndefinidas
from matrizes.matrizes import MatrizesSolver
from simulacoes.simulacoes import SimulacoesSolver


def matriz_constitutiva_carbono(angulo_graus, e1, e2, nu12, g12):
    """Matriz reduzida Qbar de uma lamina ortotropica a 0 ou 90 graus."""
    nu21 = nu12 * e2 / e1
    denominador = 1.0 - nu12 * nu21
    q11 = e1 / denominador
    q22 = e2 / denominador
    q12 = nu12 * e2 / denominador
    q66 = g12

    if math.isclose(angulo_graus % 180.0, 0.0, abs_tol=1e-12):
        return np.array(
            [[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, q66]],
            dtype=float,
        )
    if math.isclose(angulo_graus % 180.0, 90.0, abs_tol=1e-12):
        return np.array(
            [[q22, q12, 0.0], [q12, q11, 0.0], [0.0, 0.0, q66]],
            dtype=float,
        )
    raise ValueError("Esta demonstracao aceita laminas a 0 ou 90 graus.")


def transformar_tensao_para_fibras(tensao_global, angulo_graus):
    sx, sy, txy = tensao_global
    theta = math.radians(angulo_graus)
    m = math.cos(theta)
    n = math.sin(theta)
    return np.array(
        [
            m * m * sx + n * n * sy + 2.0 * m * n * txy,
            n * n * sx + m * m * sy - 2.0 * m * n * txy,
            -m * n * sx + m * n * sy + (m * m - n * n) * txy,
        ]
    )


def indice_tsai_hill(tensao_local, resistencias):
    s1, s2, t12 = tensao_local
    x_allow = resistencias["X_t" if s1 >= 0.0 else "X_c"]
    y_allow = resistencias["Y_t" if s2 >= 0.0 else "Y_c"]
    shear_allow = resistencias["S"]
    return (
        (s1 / x_allow) ** 2
        - (s1 * s2) / (x_allow**2)
        + (s2 / y_allow) ** 2
        + (t12 / shear_allow) ** 2
    )


def executar_simulacao():
    constantes = Constantes()
    equacoes = EquacoesSolver()
    edo = EDOsSolver()
    derivadas = DerivadasSolver(h=1e-5)
    matrizes = MatrizesSolver()
    simulacoes = SimulacoesSolver()
    primitivas = IntegralSolverIndefinidas()
    integrais = IntegralSolverDefinidas()

    # Condicoes de voo: altitude, atmosfera exponencial e Mach subsônico.
    altitude_m = 2000.0 * 0.3048
    rho_ar = equacoes.DensidadeBarometrica(1.225, altitude_m, 8500.0)
    temperatura_ar = 284.2
    gamma_ar = 1.4
    massa_molar_ar = 0.0289652
    r_ar = constantes.R_GASES / massa_molar_ar
    velocidade_som = math.sqrt(gamma_ar * r_ar * temperatura_ar)
    mach_voo = 0.70
    velocidade_inicial = mach_voo * velocidade_som
    pressao_dinamica_inicial = 0.5 * rho_ar * velocidade_inicial**2

    # Casco de referencia: cilindro fechado, casca fina, laminado [0/90/90/0].
    comprimento = 2.4
    raio = 0.45
    espessura_carbono = 3.5e-3
    espessura_niquel = 1.0e-3
    espessura_total = espessura_carbono + espessura_niquel
    orientacoes_carbono = (0.0, 90.0, 90.0, 0.0)
    espessura_lamina = espessura_carbono / len(orientacoes_carbono)

    # Propriedades ilustrativas; substitua pelas fichas do material fabricado.
    carbono = {
        "e1": 135e9,
        "e2": 10e9,
        "nu12": 0.30,
        "g12": 5e9,
    }
    resistencias_carbono = {
        "X_t": 1500e6,
        "X_c": 1000e6,
        "Y_t": 40e6,
        "Y_c": 150e6,
        "S": 70e6,
    }
    niquel = {"E": 200e9, "nu": 0.29, "escoamento": 1000e6}

    q_carbono = [
        matriz_constitutiva_carbono(angulo, **carbono)
        for angulo in orientacoes_carbono
    ]
    q_niquel = niquel["E"] / (1.0 - niquel["nu"] ** 2) * np.array(
        [
            [1.0, niquel["nu"], 0.0],
            [niquel["nu"], 1.0, 0.0],
            [0.0, 0.0, (1.0 - niquel["nu"]) / 2.0],
        ]
    )

    rigidez_extensional = sum(
        (q_lamina * espessura_lamina for q_lamina in q_carbono),
        start=np.zeros((3, 3)),
    ) + q_niquel * espessura_niquel

    # N = A * deformacao; pressao de 1 Pa gera os resultantes [pR/2, pR, 0].
    deformacao_por_pascal = np.asarray(
        matrizes.SistemaLinear(
            rigidez_extensional,
            [raio / 2.0, raio, 0.0],
        ),
        dtype=float,
    ).reshape(3)
    tensao_carbono_por_pascal = [
        q_lamina @ deformacao_por_pascal for q_lamina in q_carbono
    ]
    tensao_niquel_por_pascal = q_niquel @ deformacao_por_pascal

    # Cp = 1 - 4 sen^2(theta), solucao potencial para cilindro infinito.
    def coeficiente_pressao(theta):
        return 1.0 - 4.0 * math.sin(theta) ** 2

    def tensao_niquel_vm_por_pascal(theta):
        pressao_unitaria = coeficiente_pressao(theta)
        sx, sy, txy = tensao_niquel_por_pascal * pressao_unitaria
        return math.sqrt(max(0.0, sx * sx - sx * sy + sy * sy + 3.0 * txy * txy))

    def tsai_hill_por_pascal(theta):
        pressao_unitaria = coeficiente_pressao(theta)
        return max(
            indice_tsai_hill(
                transformar_tensao_para_fibras(tensao * pressao_unitaria, angulo),
                resistencias_carbono,
            )
            for tensao, angulo in zip(tensao_carbono_por_pascal, orientacoes_carbono)
        )

    # A derivada local e uma verificacao numerica do campo aerodinamico.
    gradiente_cp = derivadas.DerivadaCentralizada(coeficiente_pressao, math.pi / 4.0)

    # Integral exata do Cp medio usando as primitivas disponiveis no OMEGA.
    angulo_final = 2.0 * constantes.PI
    primitiva_cp = lambda theta: (
        integrais.RegraDaConstante(-1.0, theta)
        + 2.0 * primitivas.TrigonometricaCosseno(2.0, theta, 0.0)
    )
    cp_medio = (
        primitiva_cp(angulo_final) - primitiva_cp(0.0)
    ) / angulo_final

    # Evolucao da velocidade sob arrasto aerodinamico por RK4 do projeto.
    massa_veiculo = 1200.0
    area_frontal = 1.2
    coeficiente_arrasto = 0.35
    duracao = 2.0
    passos_tempo = 40
    passo_tempo = duracao / passos_tempo
    tempos, velocidades = edo.Rk4(
        lambda _tempo, velocidade: equacoes.AceleracaoArrastoAerodinamico(
            coeficiente_arrasto,
            area_frontal,
            massa_veiculo,
            rho_ar,
            velocidade,
        ),
        velocidade_inicial,
        passo_tempo,
        passos_tempo,
    )

    theta_grid = np.linspace(0.0, angulo_final, 73)
    x_grid = np.linspace(0.0, comprimento, 31)
    x_mesh, theta_mesh = np.meshgrid(x_grid, theta_grid, indexing="ij")
    y_mesh = raio * np.cos(theta_mesh)
    z_mesh = raio * np.sin(theta_mesh)
    cp_mesh = 1.0 - 4.0 * np.sin(theta_mesh) ** 2
    pressao_mesh = pressao_dinamica_inicial * cp_mesh
    vm_niquel_mesh = np.vectorize(tensao_niquel_vm_por_pascal)(theta_mesh) * (
        pressao_dinamica_inicial
    )
    indice_carbono_mesh = np.vectorize(tsai_hill_por_pascal)(theta_mesh) * (
        pressao_dinamica_inicial**2
    )
    utilizacao_niquel_mesh = vm_niquel_mesh / niquel["escoamento"]
    indice_global_mesh = np.maximum(
        utilizacao_niquel_mesh, np.sqrt(np.maximum(indice_carbono_mesh, 0.0))
    )

    # Media superficial de von Mises no niquel por Monte Carlo multidimensional.
    area_superficie = 2.0 * constantes.PI * raio * comprimento
    integral_vm = simulacoes.MonteCarloIntegracaoMultiD(
        lambda x, theta: (
            tensao_niquel_vm_por_pascal(theta)
            * pressao_dinamica_inicial
            * raio
        ),
        [(0.0, comprimento), (0.0, angulo_final)],
        num_amostras=12000,
        semente=20261011,
    )
    vm_medio = integral_vm["integral"] / area_superficie

    # Relatorio compacto: o maximo de carga ocorre no inicio da desaceleracao.
    arrasto_inicial = equacoes.AceleracaoArrastoAerodinamico(
        coeficiente_arrasto,
        area_frontal,
        massa_veiculo,
        rho_ar,
        velocidade_inicial,
    )
    print("TRIAGEM 3D DE CASCO CILINDRICO: CARBONO/EPOXI + LIGA DE NIQUEL")
    print("Modelo ilustrativo de casca fina; nao equivale a FEM ou certificacao.")
    print(f"Altitude: {altitude_m:.1f} m | densidade aproximada: {rho_ar:.4f} kg/m^3")
    print(
        f"Mach: {mach_voo:.2f} | velocidade inicial: {velocidade_inicial:.2f} m/s "
        f"| q inicial: {pressao_dinamica_inicial / 1000.0:.2f} kPa"
    )
    print(
        f"Casco: L={comprimento:.2f} m, R={raio:.2f} m, "
        f"t={espessura_total * 1000.0:.2f} mm; laminado carbono {orientacoes_carbono} "
        f"+ niquel {espessura_niquel * 1000.0:.2f} mm"
    )
    print(f"Arrasto inicial estimado: {arrasto_inicial:.2f} m/s^2")
    print(f"Cp medio integrado na circunferencia: {cp_medio:.4f}")
    print(f"dCp/dtheta em 45 graus (diferenca central): {gradiente_cp:.4f} rad^-1")
    print(f"Von Mises medio no niquel (Monte Carlo): {vm_medio / 1e6:.3f} MPa")
    print(
        f"Pico indice Tsai-Hill do carbono: {float(np.max(indice_carbono_mesh)):.6g} "
        "(triagem: < 1 abaixo dos allowables escolhidos)"
    )
    print(
        f"Pico utilizacao de escoamento do niquel: "
        f"{float(np.max(utilizacao_niquel_mesh)):.6g} "
        "(triagem: < 1 abaixo do escoamento assumido)"
    )
    print("Resposta transiente de velocidade e tensao equivalente no niquel:")
    for indice in np.linspace(0, passos_tempo, 5, dtype=int):
        velocidade = float(velocidades[indice])
        q_atual = 0.5 * rho_ar * velocidade**2
        vm_pico = float(np.max(vm_niquel_mesh)) * q_atual / pressao_dinamica_inicial
        print(
            f"  t={tempos[indice]:.2f} s | V={velocidade:.2f} m/s "
            f"| Mach={velocidade / velocidade_som:.3f} | pico VM={vm_pico / 1e6:.3f} MPa"
        )

    # Renderiza a superficie completa; o fechamento da malha evita uma fenda angular.
    face_values = np.concatenate((indice_global_mesh, indice_global_mesh[:, :1]), axis=1)
    x_plot = np.concatenate((x_mesh, x_mesh[:, :1]), axis=1)
    y_plot = np.concatenate((y_mesh, y_mesh[:, :1]), axis=1)
    z_plot = np.concatenate((z_mesh, z_mesh[:, :1]), axis=1)
    normalizador = colors.Normalize(vmin=0.0, vmax=max(float(face_values.max()), 1e-12))
    figura = plt.figure(figsize=(11, 8))
    eixo = figura.add_subplot(111, projection="3d")
    superficie = eixo.plot_surface(
        x_plot,
        y_plot,
        z_plot,
        facecolors=cm.viridis(normalizador(face_values)),
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    barra = cm.ScalarMappable(norm=normalizador, cmap="viridis")
    barra.set_array(face_values)
    figura.colorbar(barra, ax=eixo, shrink=0.72, label="Indice de utilizacao")
    eixo.set_xlabel("Eixo do casco x (m)")
    eixo.set_ylabel("y (m)")
    eixo.set_zlabel("z (m)")
    eixo.set_title("Casco cilindrico 3D: utilizacao local estimada")
    eixo.set_box_aspect((comprimento, 2.0 * raio, 2.0 * raio))
    destino = Path(__file__).with_name("casco_stress_3d.png")
    figura.savefig(destino, dpi=160, bbox_inches="tight")
    print(f"Visualizacao 3D salva em: {destino}")
    if "agg" not in plt.get_backend().lower():
        plt.show()

    if mach_voo >= 1.0 or not (rho_ar > 0.0 and np.isfinite(rho_ar)):
        raise AssertionError("O caso de voo deve ser finito, com Mach subsônico.")
    if rigidez_extensional.shape != (3, 3) or not np.all(np.isfinite(indice_global_mesh)):
        raise AssertionError("A resposta estrutural precisa ser finita e tridimensional.")


if __name__ == "__main__":
    executar_simulacao()