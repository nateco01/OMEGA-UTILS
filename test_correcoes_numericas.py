import math
import unittest

import numpy as np
import tensorflow as tf

from equacoes.equacoes import EDPsSolver, EDOsSolver, EquacoesSolver
from derivadas.derivadas import DerivadasSolver
from funcoes.funcoes import FuncoesSolver, FuncoesSolverNaoLinear
from integrais.integrais import IntegralSolverDefinidas
from matrizes.matrizes import MatrizesSolver


class TestCorrecoesNumericas(unittest.TestCase):
    def setUp(self):
        self.equacoes = EquacoesSolver()
        self.funcoes = FuncoesSolver()
        self.integrais = IntegralSolverDefinidas()

    def test_densidade_relativa_usa_densidades_numericas(self):
        self.assertEqual(self.equacoes.DensidadeRelativa(500.0, 1000.0), 0.5)

    def test_forca_de_lorentz_usa_produto_vetorial(self):
        forca = self.equacoes.ForcaDeLorentz(
            2.0,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 3.0]),
            np.array([0.0, 4.0, 0.0]),
        )
        np.testing.assert_allclose(forca, [26.0, 0.0, 0.0])

    def test_simpson_integra_a_funcao_em_pontos_do_intervalo(self):
        resultado = self.integrais.RegraDeSimpson(lambda x: x**2, 0.0, 1.0, 2)
        self.assertAlmostEqual(resultado, 1.0 / 3.0, places=14)

    def test_simpson_adaptativo_informa_convergencia_e_erro(self):
        resultado = self.integrais.IntegracaoSimpsonAdaptativa(
            math.sin, 0.0, math.pi, tolerancia=1e-11
        )
        self.assertTrue(resultado["convergiu"])
        self.assertAlmostEqual(resultado["integral"], 2.0, places=10)
        self.assertLessEqual(resultado["erro_estimado"], 1e-11)

    def test_integral_de_potencia(self):
        self.assertAlmostEqual(self.integrais.Integral(2.0, 2), 8.0 / 3.0)
        self.assertAlmostEqual(self.integrais.Integral(2.0, -1), math.log(2.0))

    def test_primitivas_inversas_sao_invariantes_a_sinal_do_parametro(self):
        esperado_atan = math.atan(0.5) / 2.0
        self.assertAlmostEqual(
            self.integrais.RegraDoArcoTangenteGeral(1.0, -2.0), esperado_atan
        )
        self.assertAlmostEqual(
            self.integrais.RegraDoArcoSenoGeral(1.0, -2.0), math.asin(0.5)
        )
        self.assertAlmostEqual(
            self.funcoes.HipoerbolicaTangente(1000.0), 1.0
        )

    def test_transformada_laplace_de_potencia(self):
        self.assertAlmostEqual(self.integrais.TransformadaLaplace(2, 2.0), 0.25)

    def test_funcoes_hiperbolicas_basicas(self):
        self.assertAlmostEqual(self.funcoes.HipoerbolicaSeno(1.0), math.sinh(1.0))
        self.assertAlmostEqual(self.funcoes.HipoerbolicaCosseno(1.0), math.cosh(1.0))

    def test_derivadas_de_cinco_pontos_e_adaptativa(self):
        derivadas = DerivadasSolver(h=1e-2)
        funcao = math.exp
        x = 0.3
        self.assertAlmostEqual(derivadas.Derivada(funcao, x), math.exp(x), places=9)
        self.assertAlmostEqual(
            derivadas.DerivadaProgressiva(funcao, x), math.exp(x), places=8
        )
        self.assertAlmostEqual(
            derivadas.SegundaDerivadaCentralizada(funcao, x), math.exp(x), places=7
        )
        resultado = derivadas.DerivadaAdaptativa(
            funcao, x, tolerancia=1e-12, max_refinamentos=12
        )
        self.assertTrue(resultado["convergiu"])
        self.assertAlmostEqual(resultado["valor"], math.exp(x), places=10)

    def test_rk45_controla_erro_e_integra_estado_vetorial(self):
        solver = EDOsSolver()
        tempos, estados, diagnosticos = solver.Rk45(
            lambda _t, estado: estado,
            np.array([1.0, 2.0]),
            0.0,
            1.0,
            tolerancia_absoluta=1e-10,
            tolerancia_relativa=1e-10,
        )
        np.testing.assert_allclose(estados[-1], [math.e, 2.0 * math.e], atol=1e-9)
        self.assertAlmostEqual(tempos[-1], 1.0)
        self.assertGreater(diagnosticos["passos_aceitos"], 0)

    def test_arrasto_quadratico_opoe_se_a_velocidade(self):
        aceleracao = self.equacoes.AceleracaoArrastoAerodinamico(
            0.5, 1.0, 2.0, 1.0, -10.0
        )
        self.assertGreater(aceleracao, 0.0)

    def test_densidade_e_torricelli_rejeitam_dominio_invalido(self):
        with self.assertRaises(ValueError):
            self.equacoes.Densidade(1.0, 0.0)
        with self.assertRaises(ValueError):
            self.equacoes.EquacaoDeTorricelli(0.0, -1.0, 1.0)

    def test_movimento_harmonico_respeita_frequencia_fornecida(self):
        valor = EDOsSolver().MovimentoHarmonico(1.0, 0.0, 2.0, 0.25, 9.0, 1.0)
        self.assertAlmostEqual(valor, math.cos(0.5))

    def test_edo_separavel_composta_e_integravel(self):
        solver = EDOsSolver()
        derivada = solver.EdoMetodoSeparacaoVariaveis(
            lambda x: x, lambda y: y
        )
        _tempos, estados, _diagnosticos = solver.Rk45(
            lambda x, y: derivada(x, y),
            1.0,
            0.0,
            1.0,
            tolerancia_absoluta=1e-10,
            tolerancia_relativa=1e-10,
        )
        self.assertAlmostEqual(estados[-1], math.exp(0.5), places=9)

    def test_navier_stokes_aceita_gradientes_desconectados(self):
        x = tf.Variable(2.0, dtype=tf.float64)
        y = tf.Variable(1.0, dtype=tf.float64)
        z = tf.Variable(-1.0, dtype=tf.float64)
        aceleracoes = EDOsSolver().NavierStokesConvectiva3D(
            func_u=lambda x, _y, _z: x,
            func_v=lambda _x, y, _z: tf.zeros_like(y),
            func_w=lambda _x, _y, z: tf.zeros_like(z),
            func_p=lambda x, _y, _z: 101325.0 + 0.0 * x,
            x=x,
            y=y,
            z=z,
            rho=1.0,
            nu=1.0,
        )
        np.testing.assert_allclose(
            [float(valor.numpy()) for valor in aceleracoes], [-2.0, 0.0, 0.0], atol=1e-12
        )

    def test_navier_stokes_conservativa_aceita_gradientes_desconectados(self):
        x = tf.Variable(2.0, dtype=tf.float64)
        y = tf.Variable(1.0, dtype=tf.float64)
        z = tf.Variable(-1.0, dtype=tf.float64)
        aceleracoes = EDOsSolver().NavierStokesConservativa3D(
            func_u=lambda x, _y, _z: x,
            func_v=lambda _x, y, _z: tf.zeros_like(y),
            func_w=lambda _x, _y, z: tf.zeros_like(z),
            func_p=lambda x, _y, _z: 101325.0 + 0.0 * x,
            x=x,
            y=y,
            z=z,
            rho=1.0,
            nu=1.0,
        )
        np.testing.assert_allclose(
            [float(valor.numpy()) for valor in aceleracoes], [-4.0, 0.0, 0.0], atol=1e-12
        )


class TestMatrizesNumericas(unittest.TestCase):
    def setUp(self):
        self.solver = MatrizesSolver()

    def test_gauss_jordan_faz_pivoteamento_e_varios_rhs(self):
        a = [[0.0, 1.0], [1.0, 1.0]]
        b = [[1.0, 2.0], [2.0, 4.0]]
        solucao = self.solver.GaussJordan(a, b).numpy()
        np.testing.assert_allclose(solucao, [[1.0, 2.0], [1.0, 2.0]])

    def test_gauss_jordan_aceita_sistema_invertivel_mal_escalado(self):
        solucao = self.solver.GaussJordan(
            [[1e-20, 0.0], [0.0, 1.0]], [1e-20, 2.0]
        ).numpy()
        np.testing.assert_allclose(solucao, [1.0, 2.0], rtol=1e-12, atol=1e-12)

    def test_sistema_linear_usa_float64(self):
        solucao = self.solver.SistemaLinear(
            [[1.0, 0.0], [0.0, 3.0]], [2.0, 6.0]
        )
        self.assertEqual(solucao.dtype.name, "float64")
        np.testing.assert_allclose(solucao.numpy(), [2.0, 2.0])

    def test_divisao_matricial_resolve_sem_inversa_explicita(self):
        resultado = self.solver.OperacoesMatricial(
            "div", [[2.0, 0.0], [0.0, 6.0]], [[2.0, 0.0], [0.0, 3.0]]
        )
        np.testing.assert_allclose(resultado.numpy(), [[1.0, 0.0], [0.0, 2.0]])


class TestFerramentasNaoLinearesGenericas(unittest.TestCase):
    def setUp(self):
        self.solver = FuncoesSolverNaoLinear()

    def test_resolve_sistema_acoplado(self):
        def residuo(incognitas):
            x, y = incognitas
            return [x**2 + y**2 - 5.0, x - y - 1.0]

        resultado = self.solver.ResolverSistemaNaoLinear(
            residuo, [1.5, 0.5], tolerancia=1e-10
        )
        self.assertTrue(resultado["convergiu"], resultado["mensagem"])
        np.testing.assert_allclose(resultado["solucao"], [2.0, 1.0], atol=1e-8)

    def test_multichute_encontra_raizes_distintas_sem_promessa_completude(self):
        resultado = self.solver.BuscarRaizesMultiplas(
            lambda x: [x[0] ** 2 - 1.0],
            [[-2.0], [2.0], [0.1]],
            tolerancia=1e-9,
        )
        raizes = sorted(float(raiz[0]) for raiz in resultado["raizes_encontradas"])
        self.assertEqual(len(raizes), 2)
        np.testing.assert_allclose(raizes, [-1.0, 1.0], atol=1e-7)
        self.assertFalse(resultado["completude_garantida"])

    def test_solver_nao_linear_usa_jacobiano_unilateral_no_dominio(self):
        resultado = self.solver.ResolverSistemaNaoLinear(
            lambda x: [math.log(x[0]) - math.log(2.0)],
            [0.1],
            passo_jacobiano=1.0,
            tolerancia=1e-9,
            max_iter=100,
        )
        self.assertTrue(resultado["convergiu"], resultado["mensagem"])
        self.assertAlmostEqual(resultado["solucao"][0], 2.0, places=7)

    def test_solver_nao_linear_respeita_limite_de_iteracoes(self):
        resultado = self.solver.ResolverSistemaNaoLinear(
            lambda x: [x[0] ** 2 - 2.0], [10.0], max_iter=1
        )
        self.assertLessEqual(resultado["iteracoes"], 1)

    def test_edp_discretizada_com_contorno_dirichlet(self):
        numero_nos = 7
        passo = 1.0 / (numero_nos - 1)

        def laplaciano_interno(campo):
            return (campo[:-2] - 2.0 * campo[1:-1] + campo[2:]) / passo**2

        resultado = EDPsSolver().ResolverEDPDiscretizada(
            laplaciano_interno,
            np.zeros(numero_nos),
            [lambda campo: [campo[0]], lambda campo: [campo[-1] - 1.0]],
            tolerancia=1e-10,
        )
        self.assertTrue(resultado["convergiu"], resultado["mensagem"])
        np.testing.assert_allclose(
            resultado["campo"], np.linspace(0.0, 1.0, numero_nos), atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()