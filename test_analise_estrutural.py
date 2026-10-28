import math
import unittest

import numpy as np

from estruturas import AnaliseEstruturalSolver
from materiais import Lamina, LaminaOrtotropica, MateriaisCompositosSolver


class TestMateriaisCompositos(unittest.TestCase):
    def setUp(self):
        self.material = LaminaOrtotropica(
            e1=135e9,
            e2=10e9,
            g12=5e9,
            nu12=0.3,
            alpha1=1e-6,
            alpha2=20e-6,
            x_t=1500e6,
            x_c=1000e6,
            y_t=40e6,
            y_c=150e6,
            s=70e6,
        )

    def test_q_transformada_a_noventa_graus_troca_eixos(self):
        q = MateriaisCompositosSolver.matriz_q(self.material)
        qbar = MateriaisCompositosSolver.matriz_q_transformada(self.material, 90.0)
        self.assertAlmostEqual(qbar[0, 0], q[1, 1], delta=1e-5)
        self.assertAlmostEqual(qbar[1, 1], q[0, 0], delta=1e-5)
        self.assertAlmostEqual(qbar[0, 1], q[0, 1], delta=1e-5)

    def test_laminado_simetrico_tem_b_nulo(self):
        pilhas = [
            Lamina(self.material, angulo, 0.125e-3)
            for angulo in (0.0, 90.0, 90.0, 0.0)
        ]
        resultado = MateriaisCompositosSolver.analisar_laminado(pilhas)
        np.testing.assert_allclose(resultado["B"], np.zeros((3, 3)), atol=1e-8)

    def test_lamina_livre_aquecida_nao_tem_tensao(self):
        resultado = MateriaisCompositosSolver.analisar_laminado(
            [Lamina(self.material, 0.0, 1e-3)], delta_temperatura=50.0
        )
        lado = resultado["pilhas"][0]
        np.testing.assert_allclose(
            lado["inferior"]["tensao_local"], np.zeros(3), atol=1e-7
        )
        np.testing.assert_allclose(
            lado["superior"]["tensao_local"], np.zeros(3), atol=1e-7
        )

    def test_tsai_wu_retorna_unidade_na_resistencia_uniaxial(self):
        indice = MateriaisCompositosSolver.indice_tsai_wu(
            (self.material.x_t, 0.0, 0.0), self.material
        )
        self.assertAlmostEqual(indice, 1.0)


class TestAnaliseEstrutural(unittest.TestCase):
    def test_euler_coluna_articulada(self):
        e, momento, comprimento = 200e9, 1e-6, 2.0
        carga = AnaliseEstruturalSolver.carga_critica_euler(e, momento, comprimento)
        self.assertAlmostEqual(carga, math.pi**2 * e * momento / comprimento**2)

    def test_placa_ortotropica_reduz_a_solucao_isotropica(self):
        e, nu, espessura, a, b = 70e9, 0.3, 0.01, 1.0, 0.5
        d_flexao = e * espessura**3 / (12.0 * (1.0 - nu**2))
        matriz_d = d_flexao * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
        )
        resultado = AnaliseEstruturalSolver.flambagem_placa_ortotropica_simplesmente_apoiada(
            matriz_d, a, b, modos_comprimento=3, modos_largura=3, espessura=espessura
        )
        solucao = min(
            d_flexao * (m * math.pi / a) ** 2
            + 2.0 * (nu * d_flexao + 2.0 * d_flexao * (1.0 - nu) / 2.0)
            * (n * math.pi / b) ** 2
            + d_flexao * (n * math.pi / b) ** 4 / (m * math.pi / a) ** 2
            for m in range(1, 4)
            for n in range(1, 4)
        )
        self.assertAlmostEqual(resultado["resultante_critica"], solucao, places=6)

    def test_goodman_basquin_e_miner(self):
        amplitude_eq = AnaliseEstruturalSolver.amplitude_goodman(100e6, 100e6, 500e6)
        self.assertAlmostEqual(amplitude_eq, 125e6)
        vida = AnaliseEstruturalSolver.vida_fadiga_basquin(0.5, 1.0, -0.1)
        self.assertAlmostEqual(vida, 512.0)
        dano = AnaliseEstruturalSolver.dano_acumulado_miner(
            [100.0, 200.0], [1000.0, 1000.0]
        )
        self.assertAlmostEqual(dano["dano"], 0.3)
        self.assertFalse(dano["falha_prevista"])

    def test_crescimento_paris_para_m_igual_a_dois(self):
        a0, acritica = 1e-3, 2e-3
        delta_sigma, coeficiente, y, ciclos = 100e6, 1e-20, 1.0, 1000.0
        taxa = coeficiente * (y * delta_sigma * math.sqrt(math.pi)) ** 2
        esperado = a0 * math.exp(taxa * ciclos)
        resultado = AnaliseEstruturalSolver.crescimento_trinca_paris(
            a0, acritica, delta_sigma, coeficiente, 2.0, ciclos, y
        )
        self.assertAlmostEqual(resultado["comprimento_final"], esperado)
        self.assertFalse(resultado["atingiu_critico"])

    def test_tensao_termica_totalmente_restrita(self):
        tensao = AnaliseEstruturalSolver.tensao_termica_uniaxial_restrita(
            200e9, 12e-6, 50.0
        )
        self.assertAlmostEqual(tensao, -120e6)


if __name__ == "__main__":
    unittest.main()