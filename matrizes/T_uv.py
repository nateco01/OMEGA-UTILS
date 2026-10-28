"""Integracao das equacoes TOV em unidades geometricas G=c=1.

O estado integrado e [massa, pressao, potencial_metrico] em funcao do raio.
O passo radial e fixo RK4; a superficie e localizada por bissecao com
reintegracao de subpassos RK4 no intervalo que cruza pressao zero. Reduza h
para verificar convergencia do erro global de integracao.
"""

import math

import numpy as np


class T_uv:
    def __init__(
        self,
        p_c,
        h,
        r_0,
        phi,
        p0,
        K,
        fixed,
        t_sim,
        gamma_eos=2.0,
        densidade_fixa=None,
    ):
        """Configura uma estrela esferica estatica pelo sistema TOV.

        p_c: pressao central (>0), em unidades geometricas.
        h: passo radial RK4 (>0); t_sim passos no maximo.
        r_0: raio inicial pequeno, mas positivo.
        phi: potencial inicial.
        p0: densidade de energia constante para fixed=True (compatibilidade:
            se None, usa p_c como densidade constante).
        K, gamma_eos: EOS politropica rho0=(p/K)^(1/gamma),
            epsilon=rho0+p/(gamma-1), usada quando fixed=False.
        densidade_fixa: opcional, substitui p0 no modo de densidade constante.
        """
        self.p_c = float(p_c)
        self.h = float(h)
        self.r_0 = float(r_0)
        self.phi = float(phi)
        self.fixed = fixed
        self.t_sim = t_sim
        self.gamma_eos = float(gamma_eos)
        self.K = None if K is None else float(K)

        parametros = (self.p_c, self.h, self.r_0, self.phi, self.gamma_eos)
        if not all(math.isfinite(valor) for valor in parametros):
            raise ValueError("Os parametros TOV devem ser finitos.")
        if self.p_c <= 0.0 or self.h <= 0.0 or self.r_0 <= 0.0:
            raise ValueError("p_c, h e r_0 devem ser positivos.")
        if not isinstance(fixed, bool):
            raise TypeError("fixed deve ser bool.")
        if isinstance(t_sim, bool) or not isinstance(t_sim, int) or t_sim < 1:
            raise ValueError("t_sim deve ser um inteiro positivo (numero maximo de passos).")

        if fixed:
            densidade = densidade_fixa if densidade_fixa is not None else p0
            if densidade is None:
                densidade = self.p_c
            self.densidade_fixa = float(densidade)
            if not math.isfinite(self.densidade_fixa) or self.densidade_fixa <= 0.0:
                raise ValueError("A densidade de energia fixa deve ser positiva e finita.")
        else:
            if self.K is None or not math.isfinite(self.K) or self.K <= 0.0:
                raise ValueError("K deve ser positivo e finito para a EOS politropica.")
            if self.gamma_eos <= 1.0:
                raise ValueError("gamma_eos deve ser maior que 1.")

        energia_central = self.e_p_metodo(self.p_c)
        self.m_r0 = (4.0 * math.pi / 3.0) * energia_central * self.r_0**3
        if self.r_0 <= 2.0 * self.m_r0:
            raise ValueError("A condicao inicial esta dentro do raio de Schwarzschild local.")
        self.lista = []
        self.lista_r = []
        self.superficie_alcancada = False

    def p_0_metodo(self, K=None, pressao=None):
        """Densidade de repouso da EOS politropica no valor de pressao dado."""
        if self.fixed:
            return self.densidade_fixa
        constante = self.K if K is None else float(K)
        pressao_local = self.p_c if pressao is None else float(pressao)
        if constante <= 0.0 or pressao_local < 0.0:
            raise ValueError("K deve ser positivo e a pressao nao negativa.")
        return (pressao_local / constante) ** (1.0 / self.gamma_eos)

    def e_p_metodo(self, pressao=None):
        """Densidade de energia epsilon(p) pela EOS configurada."""
        if self.fixed:
            return self.densidade_fixa
        pressao_local = self.p_c if pressao is None else float(pressao)
        if not math.isfinite(pressao_local) or pressao_local < 0.0:
            raise ValueError("A pressao da EOS deve ser finita e nao negativa.")
        rho0 = self.p_0_metodo(pressao=pressao_local)
        return rho0 + pressao_local / (self.gamma_eos - 1.0)

    def y(self):
        """Estado inicial [massa, pressao, potencial] no raio r_0."""
        return np.array([self.m_r0, self.p_c, self.phi], dtype=float)

    def _derivadas_tov(self, raio, estado):
        massa, pressao, _potencial = estado
        if not all(math.isfinite(float(valor)) for valor in estado):
            raise FloatingPointError("O estado TOV contem valores nao finitos.")
        if raio <= 0.0 or raio <= 2.0 * massa:
            raise ValueError("A integracao chegou a r <= 2m, onde as equacoes TOV sao singulares.")
        if pressao <= 0.0:
            return np.zeros(3, dtype=float)

        energia = self.e_p_metodo(pressao)
        denominador = raio * (raio - 2.0 * massa)
        termo_gravitacional = massa + 4.0 * math.pi * raio**3 * pressao
        return np.array(
            [
                4.0 * math.pi * raio**2 * energia,
                -(energia + pressao) * termo_gravitacional / denominador,
                termo_gravitacional / denominador,
            ],
            dtype=float,
        )

    def _passo_rk4(self, raio, estado, passo):
        k1 = self._derivadas_tov(raio, estado)
        k2 = self._derivadas_tov(raio + passo / 2.0, estado + passo * k1 / 2.0)
        k3 = self._derivadas_tov(raio + passo / 2.0, estado + passo * k2 / 2.0)
        k4 = self._derivadas_tov(raio + passo, estado + passo * k3)
        return estado + (passo / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def Rk4(self):
        """Integra os estados TOV e para ao atingir a superficie p=0."""
        raio = self.r_0
        estado = self.y()
        self.lista = [estado.copy()]
        self.lista_r = [raio]
        self.superficie_alcancada = False

        for _ in range(self.t_sim):
            proximo = self._passo_rk4(raio, estado, self.h)
            proximo_raio = raio + self.h
            if not np.all(np.isfinite(proximo)):
                raise FloatingPointError("O passo RK4 produziu um estado TOV nao finito.")

            if proximo[1] <= 0.0:
                inferior, superior = 0.0, 1.0
                for _ in range(48):
                    fracao = 0.5 * (inferior + superior)
                    tentativa = self._passo_rk4(raio, estado, fracao * self.h)
                    if tentativa[1] > 0.0:
                        inferior = fracao
                    else:
                        superior = fracao
                fracao = 0.5 * (inferior + superior)
                estado = self._passo_rk4(raio, estado, fracao * self.h)
                estado[1] = 0.0
                raio += fracao * self.h
                self.superficie_alcancada = True
                self.lista.append(estado.copy())
                self.lista_r.append(raio)
                break

            estado = proximo
            raio = proximo_raio
            self.lista.append(estado.copy())
            self.lista_r.append(raio)

        return np.asarray(self.lista, dtype=float)

    def plot_dados(self):
        import matplotlib.pyplot as plt

        if not self.lista:
            raise ValueError("Execute Rk4() antes de plotar os resultados TOV.")
        estados = np.asarray(self.lista, dtype=float)
        raios = np.asarray(self.lista_r, dtype=float)
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        axs[0].plot(raios, estados[:, 0], color="blue", label="massa")
        axs[0].set_ylabel("Massa (unidades geometricas)")
        axs[1].plot(raios, estados[:, 1], color="red", label="pressao")
        axs[1].set_ylabel("Pressao")
        axs[2].plot(raios, estados[:, 2], color="green", label="potencial")
        axs[2].set_ylabel("Potencial")
        axs[2].set_xlabel("Raio")
        for eixo in axs:
            eixo.grid(True, linestyle="--", alpha=0.7)
            eixo.legend()
        fig.suptitle("Integracao das equacoes TOV")
        fig.tight_layout()
        plt.show()