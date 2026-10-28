import unittest

import numpy as np

from matrizes.T_uv import T_uv


class TestTOV(unittest.TestCase):
    def test_integracao_evolui_estado_fisico(self):
        solver = T_uv(
            p_c=0.1,
            h=1e-3,
            r_0=0.01,
            phi=0.0,
            p0=1.0,
            K=None,
            fixed=True,
            t_sim=10,
        )
        estados = solver.Rk4()
        self.assertEqual(estados.shape, (11, 3))
        self.assertTrue(np.all(np.diff(estados[:, 0]) > 0.0))
        self.assertTrue(np.all(np.diff(estados[:, 1]) < 0.0))
        self.assertEqual(len(solver.lista_r), len(estados))

    def test_eos_politropica_usa_gamma_configurado(self):
        solver = T_uv(
            p_c=0.1,
            h=1e-3,
            r_0=0.01,
            phi=0.0,
            p0=None,
            K=0.1,
            fixed=False,
            t_sim=2,
            gamma_eos=2.0,
        )
        self.assertAlmostEqual(solver.p_0_metodo(), 1.0)
        self.assertAlmostEqual(solver.e_p_metodo(), 1.1)

    def test_rejeita_estado_inicial_dentro_do_horizonte(self):
        with self.assertRaises(ValueError):
            T_uv(
                p_c=1.0,
                h=0.1,
                r_0=0.01,
                phi=0.0,
                p0=1e9,
                K=None,
                fixed=True,
                t_sim=5,
            )

    def test_raio_da_superficie_converge_ao_refinar_passo(self):
        raios = []
        for passo in (0.02, 0.01, 0.005):
            solver = T_uv(
                p_c=0.1,
                h=passo,
                r_0=0.01,
                phi=0.0,
                p0=1.0,
                K=None,
                fixed=True,
                t_sim=100,
            )
            solver.Rk4()
            self.assertTrue(solver.superficie_alcancada)
            self.assertEqual(solver.lista[-1][1], 0.0)
            raios.append(solver.lista_r[-1])
        self.assertLess(abs(raios[1] - raios[2]), abs(raios[0] - raios[1]))


if __name__ == "__main__":
    unittest.main()