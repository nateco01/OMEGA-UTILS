import math
import unittest

import numpy as np

from derivadas.derivadas import DerivadasSolver
from equacoes.equacoes import EDPsSolver
from geometria_diferencial.Schwarzschild import Schwarzschild


class TestEquacoesRicciVacuo(unittest.TestCase):
    def setUp(self):
        self.solver = EDPsSolver()
        self.derivadas = DerivadasSolver(h=1e-4)
        self.coordenadas = (0.2, -0.3, 0.4, 0.1)

    def test_simbolos_nulos_produzem_ricci_nulo(self):
        ricci = self.solver.TensorRicci(
            lambda *_coords: np.zeros((4, 4, 4)),
            self.coordenadas,
            self.derivadas,
        )
        np.testing.assert_array_equal(ricci, np.zeros((4, 4)))

    def test_derivada_conhecida_produz_r00_unitario(self):
        def gamma(x0, x1, x2, x3):
            tensor = np.zeros((4, 4, 4))
            tensor[1, 0, 0] = x1
            return tensor

        ricci = self.solver.TensorRicci(gamma, self.coordenadas, self.derivadas)
        self.assertAlmostEqual(ricci[0, 0], 1.0, places=8)
        np.testing.assert_allclose(ricci, ricci.T, atol=1e-8)

    def test_api_dez_residuos_retorna_componentes_independentes(self):
        residuos = self.solver.EquacoesRicciVacio(
            lambda *_coords: np.zeros((4, 4, 4)),
            self.coordenadas,
            self.derivadas,
        )
        self.assertEqual(len(residuos), 10)
        self.assertEqual(set(residuos), {(mu, nu) for mu in range(4) for nu in range(mu, 4)})
        self.assertTrue(all(valor == 0.0 for valor in residuos.values()))

    def test_aceita_callable_por_componente(self):
        def gamma(rho, mu, nu, x0, x1, x2, x3):
            if (rho, mu, nu) == (1, 0, 0):
                return x1
            return 0.0

        ricci = self.solver.TensorRicci(gamma, self.coordenadas, self.derivadas)
        self.assertAlmostEqual(ricci[0, 0], 1.0, places=8)

    def test_metrica_cilindrica_de_espaco_plano_tem_ricci_nulo(self):
        def metrica(_tempo, raio, _theta, _z):
            return np.diag([-1.0, 1.0, raio**2, 1.0])

        coordenadas = (0.0, 2.0, 0.4, 0.0)
        gamma = self.solver.SimbolosChristoffel(
            metrica, coordenadas, DerivadasSolver(h=1e-3)
        )
        self.assertAlmostEqual(gamma[1, 2, 2], -2.0, places=7)
        self.assertAlmostEqual(gamma[2, 1, 2], 0.5, places=7)
        residuos = self.solver.EquacoesRicciVacioDaMetrica(
            metrica, coordenadas, DerivadasSolver(h=1e-3)
        )
        self.assertLess(max(abs(valor) for valor in residuos.values()), 1e-6)

    def test_ajusta_parametro_de_metrica_com_dado_de_contorno(self):
        def modelo(parametros, _tempo, _x, _y, _z):
            return np.diag([-1.0, parametros[0] ** 2, 1.0, 1.0])

        resultado = self.solver.ResolverMetricaVacio(
            modelo_metrica=modelo,
            parametros_iniciais=[1.0],
            pontos_colocacao=[(0.0, 1.0, 0.0, 0.0)],
            condicoes_dados=[lambda parametros: parametros[0] - 2.0],
            max_iter=30,
        )
        self.assertTrue(resultado["convergiu"], resultado["mensagem"])
        self.assertAlmostEqual(resultado["parametros"][0], 2.0, places=8)
        self.assertLess(resultado["norma_residuo"], 1e-8)

    def test_rejeita_coordenadas_invalidas(self):
        with self.assertRaises(ValueError):
            self.solver.TensorRicci(lambda *_coords: np.zeros((4, 4, 4)), (0, 1, 2))

    def test_rk4_mantem_orbita_circular_estavel(self):
        raio = 10.0
        momento_angular = math.sqrt(raio**2 / (raio - 3.0))
        orbita = Schwarzschild(
            r_0=raio,
            v_0=0.0,
            L=momento_angular,
            h=1e-3,
            N=10,
            phi=0.0,
            t=0.0,
            tempo_sim=0.01,
        )
        orbita.Rk4()
        estado_final = np.asarray(orbita.lista[-1], dtype=float)
        self.assertAlmostEqual(estado_final[0], raio, places=10)
        self.assertAlmostEqual(estado_final[1], 0.0, places=10)

    def test_curvatura_escalar_e_einstein_da_esfera_unitaria(self):
        def esfera(theta, _phi):
            return np.diag([1.0, math.sin(theta) ** 2])

        ponto = (0.8, 0.3)
        derivadas = DerivadasSolver(h=1e-3)
        riemann = self.solver.TensorRiemannDaMetrica(esfera, ponto, derivadas)
        ricci = self.solver.TensorRicciDaMetrica(esfera, ponto, derivadas)

        self.assertAlmostEqual(
            riemann[0, 1, 0, 1], math.sin(ponto[0]) ** 2, places=7
        )
        np.testing.assert_allclose(ricci, esfera(*ponto), atol=2e-7)
        self.assertAlmostEqual(self.solver.EscalarRicci(esfera, ponto, derivadas), 2.0, places=7)
        np.testing.assert_allclose(
            self.solver.TensorEinstein(esfera, ponto, derivadas),
            np.zeros((2, 2)),
            atol=2e-7,
        )

    def test_vacuo_de_schwarzschild_tem_curvatura_nao_nula_e_ricci_nulo(self):
        def schwarzschild(_tempo, raio, theta, _phi):
            f = 1.0 - 2.0 / raio
            return np.diag(
                [-f, 1.0 / f, raio**2, raio**2 * math.sin(theta) ** 2]
            )

        ponto = (0.0, 10.0, 0.8, 0.0)
        riemann = self.solver.TensorRiemannDaMetrica(
            schwarzschild, ponto, DerivadasSolver(h=1e-3)
        )
        ricci = np.einsum("rmrn->mn", riemann)

        self.assertGreater(float(np.max(np.abs(riemann))), 1e-2)
        self.assertLess(float(np.max(np.abs(ricci))), 5e-8)

    def test_derivada_covariante_preserva_a_metrica(self):
        def esfera(theta, _phi):
            return np.diag([1.0, math.sin(theta) ** 2])

        derivadas = DerivadasSolver(h=1e-3)
        resultado = self.solver.DerivadaCovariante(
            esfera,
            esfera,
            (0.8, 0.3),
            ("covariante", "covariante"),
            derivadas,
        )
        np.testing.assert_allclose(resultado, np.zeros((2, 2, 2)), atol=2e-10)
        derivada_vetor = self.solver.DerivadaCovariante(
            esfera,
            lambda _theta, _phi: np.array([1.0, 0.0]),
            (0.8, 0.3),
            ("contravariante",),
            derivadas,
        )
        self.assertAlmostEqual(derivada_vetor[1, 1], 1.0 / math.tan(0.8), places=7)

    def test_geodesica_minkowski_e_linear_e_preserva_norma(self):
        def minkowski(_tempo, _x):
            return np.diag([-1.0, 1.0])

        resultado = self.solver.Geodesica(
            minkowski,
            coordenadas_iniciais=(0.0, 1.0),
            velocidades_iniciais=(1.0, 0.25),
            parametro_inicial=0.0,
            parametro_final=2.0,
            derivadas=DerivadasSolver(h=1e-3),
            passo_inicial=0.2,
        )
        np.testing.assert_allclose(
            resultado["coordenadas"][-1], [2.0, 1.5], atol=1e-10
        )
        np.testing.assert_allclose(
            resultado["norma_velocidade"], [-1.0 + 0.25**2] * len(resultado["parametros"]),
            atol=1e-12,
        )

    def test_geodesica_esferica_segue_grande_circulo(self):
        def esfera(theta, _phi):
            return np.diag([1.0, math.sin(theta) ** 2])

        theta_inicial = math.pi / 3.0
        parametro_final = 0.5
        resultado = self.solver.Geodesica(
            esfera,
            coordenadas_iniciais=(theta_inicial, 0.0),
            velocidades_iniciais=(0.0, 1.0 / math.sin(theta_inicial)),
            parametro_inicial=0.0,
            parametro_final=parametro_final,
            derivadas=DerivadasSolver(h=1e-3),
            passo_inicial=0.05,
            tolerancia_absoluta=1e-10,
            tolerancia_relativa=1e-10,
        )
        esperado_theta = math.acos(
            math.cos(theta_inicial) * math.cos(parametro_final)
        )
        esperado_phi = math.atan2(
            math.sin(parametro_final),
            math.sin(theta_inicial) * math.cos(parametro_final),
        )
        np.testing.assert_allclose(
            resultado["coordenadas"][-1],
            [esperado_theta, esperado_phi],
            atol=2e-7,
        )
        self.assertAlmostEqual(resultado["norma_velocidade"][-1], 1.0, places=7)

    def test_simbolos_christoffel_e_riemann_sao_dimensionais(self):
        def polar(_raio, _angulo):
            return np.diag([1.0, _raio**2])

        ponto = (2.0, 0.4)
        gamma = self.solver.SimbolosChristoffel(
            polar, ponto, DerivadasSolver(h=1e-3)
        )
        self.assertEqual(gamma.shape, (2, 2, 2))
        self.assertAlmostEqual(gamma[0, 1, 1], -2.0, places=8)
        self.assertAlmostEqual(gamma[1, 0, 1], 0.5, places=8)
        ricci = self.solver.TensorRicciDaMetrica(
            polar, ponto, DerivadasSolver(h=1e-3)
        )
        np.testing.assert_allclose(ricci, np.zeros((2, 2)), atol=2e-7)

    def test_rejeita_metrica_singular(self):
        with self.assertRaisesRegex(ValueError, "singular"):
            self.solver.SimbolosChristoffel(
                lambda _x, _y: np.zeros((2, 2)), (0.0, 0.0)
            )


if __name__ == "__main__":
    unittest.main()