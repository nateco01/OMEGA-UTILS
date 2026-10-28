import math
from pathlib import Path

import numpy as np

from constantes.constantes import Constantes
from equacoes.equacoes import EDOsSolver
from funcoes.funcoes import FuncoesSolverNaoLinear


class SimulacaoTempoRealVenturi3D:
    """Modelo quase-1D compressível em um bocal axisimétrico.

    O campo 3D é obtido estendendo o escoamento médio pelas seções circulares
    e calculando a componente radial necessária para acompanhar a parede.
    O modelo supõe gás ideal caloricamente perfeito, escoamento adiabático e
    sem atrito. Não substitui uma simulação CFD viscosa/turbulenta.
    """

    def __init__(
        self,
        p0=500000.0,
        T0=900.0,
        gamma=1.4,
        massa_molar=0.0289652,
        p_contrapressao=None,
        L_total=2.0,
        A_entrada=0.1,
        A_garganta=0.02,
    ):
        self.const = Constantes()
        self.nl = FuncoesSolverNaoLinear()
        self.edo = EDOsSolver()

        self.p0 = float(p0)
        self.T0 = float(T0)
        self.gamma = float(gamma)
        self.R_gas = self.const.R_GASES / float(massa_molar)
        self.p_contrapressao = (
            self.const.P_ATM if p_contrapressao is None else float(p_contrapressao)
        )
        self.L_total = float(L_total)
        self.A_entrada = float(A_entrada)
        self.A_garganta = float(A_garganta)
        self.x_garganta = self.L_total / 2.0

        self._validar_parametros()
        self.A_saida = self.area_transversal(self.L_total)
        self._definir_regime()

    def _validar_parametros(self):
        if not math.isfinite(self.p0) or self.p0 <= 0:
            raise ValueError("p0 deve ser uma pressão absoluta positiva, em Pa.")
        if not math.isfinite(self.T0) or self.T0 <= 0:
            raise ValueError("T0 deve ser uma temperatura absoluta positiva, em K.")
        if not math.isfinite(self.gamma) or self.gamma <= 1:
            raise ValueError("gamma deve ser maior que 1.")
        if not math.isfinite(self.R_gas) or self.R_gas <= 0:
            raise ValueError("A massa molar deve ser positiva.")
        if not 0 < self.p_contrapressao < self.p0:
            raise ValueError("A contrapressão deve estar entre zero e p0.")
        if not math.isfinite(self.L_total) or self.L_total <= 0:
            raise ValueError("O comprimento do bocal deve ser positivo.")
        if (
            not math.isfinite(self.A_entrada)
            or not math.isfinite(self.A_garganta)
            or self.A_entrada <= self.A_garganta
            or self.A_garganta <= 0
        ):
            raise ValueError(
                "As áreas devem ser positivas e A_entrada deve ser maior que A_garganta."
            )

    def area_transversal(self, x, L_total=None, A_entrada=None, A_garganta=None):
        """Área circular por trecho linear, com garganta no meio do bocal."""
        comprimento = self.L_total if L_total is None else float(L_total)
        area_entrada = self.A_entrada if A_entrada is None else float(A_entrada)
        area_garganta = (
            self.A_garganta if A_garganta is None else float(A_garganta)
        )
        x_limitado = min(max(float(x), 0.0), comprimento)
        meio = comprimento / 2.0

        if x_limitado <= meio:
            return area_entrada - (area_entrada - area_garganta) * (
                x_limitado / meio
            )
        return area_garganta + (area_entrada - area_garganta) * (
            (x_limitado - meio) / meio
        )

    def _gradiente_area(self, x):
        if x < self.x_garganta:
            return -2.0 * (self.A_entrada - self.A_garganta) / self.L_total
        return 2.0 * (self.A_entrada - self.A_garganta) / self.L_total

    def _raio(self, x):
        return math.sqrt(self.area_transversal(x) / self.const.PI)

    def _gradiente_raio(self, x):
        area = self.area_transversal(x)
        return self._gradiente_area(x) / (2.0 * math.sqrt(self.const.PI * area))

    def _razao_area_mach(self, mach):
        g = self.gamma
        termo = (2.0 / (g + 1.0)) * (
            1.0 + ((g - 1.0) / 2.0) * mach**2
        )
        return (termo ** ((g + 1.0) / (2.0 * (g - 1.0)))) / mach

    def _resolver_raiz_com_omega(self, funcao, inferior, superior, tolerancia=1e-10):
        """Usa secantes do OMEGA e mantém bisseção para garantir convergência."""
        f_inferior = funcao(inferior)
        f_superior = funcao(superior)
        if f_inferior == 0:
            return inferior
        if f_superior == 0:
            return superior
        if f_inferior * f_superior > 0:
            raise ValueError("O intervalo não contém uma raiz com mudança de sinal.")

        intervalo = superior - inferior
        candidato = self.nl.MetodoSecantes(
            funcao,
            inferior + 0.25 * intervalo,
            inferior + 0.75 * intervalo,
            tol=tolerancia,
            max_iter=80,
        )
        if not isinstance(candidato, str):
            candidato = float(candidato)
            if (
                math.isfinite(candidato)
                and inferior < candidato < superior
            ):
                f_candidato = funcao(candidato)
                if abs(f_candidato) <= tolerancia:
                    return candidato
                if f_inferior * f_candidato < 0:
                    superior, f_superior = candidato, f_candidato
                else:
                    inferior, f_inferior = candidato, f_candidato

        for _ in range(100):
            meio = (inferior + superior) / 2.0
            f_meio = funcao(meio)
            if abs(f_meio) <= tolerancia or superior - inferior <= tolerancia:
                return meio
            if f_inferior * f_meio < 0:
                superior, f_superior = meio, f_meio
            else:
                inferior, f_inferior = meio, f_meio

        raise RuntimeError("O solver do OMEGA não convergiu para a raiz.")

    def _mach_por_razao_area(self, razao_area, ramo):
        if razao_area < 1.0 - 1e-10:
            raise ValueError("A razão A/A* não pode ser menor que 1.")
        if math.isclose(razao_area, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return 1.0

        residuo = lambda mach: self._razao_area_mach(mach) - razao_area
        if ramo == "subsonico":
            inferior, superior = 1e-10, 1.0
        elif ramo == "supersonico":
            inferior, superior = 1.0, 2.0
            while residuo(superior) < 0:
                superior *= 2.0
                if superior > 64.0:
                    raise RuntimeError("Não foi possível limitar a raiz supersônica.")
        else:
            raise ValueError("O ramo deve ser 'subsonico' ou 'supersonico'.")

        return self._resolver_raiz_com_omega(residuo, inferior, superior)

    def _razao_pressao_isentropica(self, mach):
        g = self.gamma
        termo = 1.0 + ((g - 1.0) / 2.0) * mach**2
        return termo ** (-g / (g - 1.0))

    def _mach_por_pressao(self, pressao_estatica, pressao_estagnacao):
        razao = pressao_estatica / pressao_estagnacao
        if not 0 < razao < 1:
            raise ValueError("A razão entre pressões deve estar entre zero e um.")
        g = self.gamma
        termo = (pressao_estagnacao / pressao_estatica) ** (
            (g - 1.0) / g
        )
        return math.sqrt((2.0 / (g - 1.0)) * (termo - 1.0))

    def _estado_apos_choque_normal(self, x_choque):
        mach_antes = self._mach_por_razao_area(
            self.area_transversal(x_choque) / self.A_garganta,
            "supersonico",
        )
        g = self.gamma
        mach_depois_quadrado = (
            1.0 + ((g - 1.0) / 2.0) * mach_antes**2
        ) / (g * mach_antes**2 - (g - 1.0) / 2.0)
        mach_depois = math.sqrt(mach_depois_quadrado)
        salto_pressao = 1.0 + (2.0 * g / (g + 1.0)) * (
            mach_antes**2 - 1.0
        )
        pressao_antes = self.p0 * self._razao_pressao_isentropica(mach_antes)
        pressao_depois = pressao_antes * salto_pressao
        p0_depois = pressao_depois / self._razao_pressao_isentropica(mach_depois)
        area_estrela_depois = self.A_garganta * self.p0 / p0_depois
        return mach_antes, mach_depois, p0_depois, area_estrela_depois

    def _pressao_saida_com_choque(self, x_choque):
        _, _, p0_depois, area_estrela_depois = (
            self._estado_apos_choque_normal(x_choque)
        )
        mach_saida = self._mach_por_razao_area(
            self.A_saida / area_estrela_depois,
            "subsonico",
        )
        return p0_depois * self._razao_pressao_isentropica(mach_saida)

    def _definir_regime(self):
        razao_saida_garganta = self.A_saida / self.A_garganta
        mach_saida_limite = self._mach_por_razao_area(
            razao_saida_garganta,
            "subsonico",
        )
        self.p_limite_nao_estrangulado = (
            self.p0 * self._razao_pressao_isentropica(mach_saida_limite)
        )
        self.x_choque = None
        self.p0_apos_choque = None
        self.A_estrela_apos_choque = None
        self.A_estrela = self.A_garganta

        if self.p_contrapressao >= self.p_limite_nao_estrangulado:
            self.regime = "subsônico não estrangulado"
            mach_saida = self._mach_por_pressao(self.p_contrapressao, self.p0)
            self.A_estrela = self.A_saida / self._razao_area_mach(mach_saida)
            return

        mach_saida_supersonico = self._mach_por_razao_area(
            razao_saida_garganta,
            "supersonico",
        )
        self.p_saida_isentropica = (
            self.p0
            * self._razao_pressao_isentropica(mach_saida_supersonico)
        )
        p_saida_com_choque_na_saida = self._pressao_saida_com_choque(
            self.L_total
        )

        if self.p_contrapressao >= p_saida_com_choque_na_saida:
            funcao = lambda x: (
                self._pressao_saida_com_choque(x) - self.p_contrapressao
            )
            self.x_choque = self._resolver_raiz_com_omega(
                funcao,
                self.x_garganta,
                self.L_total,
                tolerancia=1e-10,
            )
            (
                _,
                _,
                self.p0_apos_choque,
                self.A_estrela_apos_choque,
            ) = self._estado_apos_choque_normal(self.x_choque)
            self.regime = "estrangulado com choque normal interno"
        else:
            self.regime = "estrangulado, supersônico até a saída"
            if self.p_contrapressao < self.p_saida_isentropica:
                self.regime += " (jato subexpandido)"
            elif self.p_contrapressao > self.p_saida_isentropica:
                self.regime += " (choque fora do bocal)"
            else:
                self.regime += " (adaptado à contrapressão)"

    def _estado_axial(self, x):
        x = min(max(float(x), 0.0), self.L_total)
        area = self.area_transversal(x)

        if self.regime == "subsônico não estrangulado":
            mach = self._mach_por_razao_area(
                area / self.A_estrela,
                "subsonico",
            )
            p0_local = self.p0
        elif self.x_choque is not None and x >= self.x_choque:
            mach = self._mach_por_razao_area(
                area / self.A_estrela_apos_choque,
                "subsonico",
            )
            p0_local = self.p0_apos_choque
        else:
            ramo = (
                "subsonico"
                if x < self.x_garganta
                else "supersonico"
            )
            mach = self._mach_por_razao_area(
                area / self.A_garganta,
                ramo,
            )
            p0_local = self.p0

        temperatura = self.T0 / (
            1.0 + ((self.gamma - 1.0) / 2.0) * mach**2
        )
        pressao = p0_local * self._razao_pressao_isentropica(mach)
        densidade = pressao / (self.R_gas * temperatura)
        velocidade_axial = mach * math.sqrt(
            self.gamma * self.R_gas * temperatura
        )
        return {
            "Mach": mach,
            "Pressao": pressao,
            "Temperatura": temperatura,
            "Densidade": densidade,
            "VelocidadeAxial": velocidade_axial,
            "Area": area,
        }

    def obter_estado_fluido(self, x, y, z):
        """Calcula o estado local e as componentes de velocidade u, v e w."""
        if not all(math.isfinite(float(coordenada)) for coordenada in (x, y, z)):
            raise ValueError("As coordenadas x, y e z devem ser finitas.")
        x = min(max(float(x), 0.0), self.L_total)
        y = float(y)
        z = float(z)
        estado = self._estado_axial(x)
        raio = self._raio(x)
        raio_local = math.hypot(y, z)
        if raio_local > raio * (1.0 + 1e-10):
            raise ValueError("O ponto informado está fora da seção do bocal.")

        velocidade_radial = (
            estado["VelocidadeAxial"]
            * (raio_local / raio)
            * self._gradiente_raio(x)
        )
        if raio_local == 0:
            velocidade_y = 0.0
            velocidade_z = 0.0
        else:
            velocidade_y = velocidade_radial * y / raio_local
            velocidade_z = velocidade_radial * z / raio_local

        return {
            "Mach": estado["Mach"],
            "Pressao": estado["Pressao"],
            "Temperatura": estado["Temperatura"],
            "Velocidade": (
                estado["VelocidadeAxial"],
                velocidade_y,
                velocidade_z,
            ),
            "Densidade": estado["Densidade"],
        }

    def _derivada_tempo(self, x, _tempo):
        return 1.0 / self._estado_axial(x)["VelocidadeAxial"]

    def rodar_tempo_real(
        self,
        pos_inicial=(0.05, 0.01, 0.005),
        passos=80,
        imprimir=True,
    ):
        """Integra uma linha de corrente 3D e o tempo de trânsito pelo bocal."""
        if not isinstance(passos, int) or passos <= 0:
            raise ValueError("passos deve ser um inteiro positivo.")
        x0, y0, z0 = map(float, pos_inicial)
        if not math.isfinite(x0) or not 0.0 <= x0 < self.L_total:
            raise ValueError("A posição inicial x deve estar dentro do bocal.")
        estado_inicial = self.obter_estado_fluido(x0, y0, z0)
        if estado_inicial["Velocidade"][0] <= 0:
            raise ValueError("A velocidade axial inicial deve ser positiva.")

        fracao_y = y0 / self._raio(x0)
        fracacao_z = z0 / self._raio(x0)
        posicoes_x = np.linspace(x0, self.L_total, passos + 1)
        tempos = [0.0]
        tempo_atual = 0.0

        for x_inicio, x_fim in zip(posicoes_x[:-1], posicoes_x[1:]):
            divisoes = [float(x_inicio), float(x_fim)]
            for descontinuidade in (self.x_garganta, self.x_choque):
                if (
                    descontinuidade is not None
                    and x_inicio < descontinuidade < x_fim
                ):
                    divisoes.append(descontinuidade)
            divisoes.sort()

            for a, b in zip(divisoes[:-1], divisoes[1:]):
                quantidade_passos = max(
                    1, math.ceil((b - a) / (self.L_total / passos))
                )
                passo = (b - a) / quantidade_passos
                _, valores_tempo = self.edo.Rk4(
                    self._derivada_tempo,
                    tempo_atual,
                    passo,
                    quantidade_passos,
                    t0=a,
                )
                tempo_atual = valores_tempo[-1]
            tempos.append(tempo_atual)

        resultados = []
        for x, tempo in zip(posicoes_x, tempos):
            raio = self._raio(x)
            y = fracao_y * raio
            z = fracacao_z * raio
            estado = self.obter_estado_fluido(x, y, z)
            resultados.append(
                {
                    "Tempo": float(tempo),
                    "Posicao": (x, y, z),
                    **estado,
                }
            )

        if imprimir:
            print(
                f"Regime: {self.regime} | "
                f"Contrapressão: {self.p_contrapressao:.1f} Pa"
            )
            if self.x_choque is not None:
                print(f"Posição estimada do choque: x = {self.x_choque:.4f} m")
            print(
                f"{'t (ms)':>9} | {'x (m)':>8} | {'y (m)':>8} | "
                f"{'z (m)':>8} | {'Mach':>7} | {'p (Pa)':>12} | {'u (m/s)':>10}"
            )
            indices = np.linspace(
                0, len(resultados) - 1, min(15, len(resultados)), dtype=int
            )
            for indice in indices:
                ponto = resultados[int(indice)]
                x, y, z = ponto["Posicao"]
                u = ponto["Velocidade"][0]
                print(
                    f"{ponto['Tempo'] * 1000:9.3f} | {x:8.4f} | "
                    f"{y:8.4f} | {z:8.4f} | {ponto['Mach']:7.4f} | "
                    f"{ponto['Pressao']:12.1f} | {u:10.2f}"
                )
            massa_vazao = (
                resultados[-1]["Densidade"]
                * resultados[-1]["Velocidade"][0]
                * self.A_saida
            )
            print(f"Vazão mássica estimada: {massa_vazao:.6f} kg/s")
            print(f"Tempo de trânsito: {resultados[-1]['Tempo'] * 1000:.3f} ms")
        return resultados

    def calcular_campo_3d(self, n_x=41, n_radial=9, n_theta=24):
        """Gera campos em uma malha polar 3D (x, raio, ângulo)."""
        if any(not isinstance(n, int) or n < 2 for n in (n_x, n_radial, n_theta)):
            raise ValueError("n_x, n_radial e n_theta devem ser inteiros >= 2.")

        x_grid = np.linspace(0.0, self.L_total, n_x)
        fracao_radial = np.linspace(0.0, 1.0, n_radial)
        theta = np.linspace(0.0, 2.0 * self.const.PI, n_theta, endpoint=False)
        shape = (n_x, n_radial, n_theta)

        raios = np.sqrt(
            np.array([self.area_transversal(x) for x in x_grid]) / self.const.PI
        )
        raio_local = raios[:, None, None] * fracao_radial[None, :, None]
        y = raio_local * np.cos(theta)[None, None, :]
        z = raio_local * np.sin(theta)[None, None, :]
        x = np.broadcast_to(x_grid[:, None, None], shape).copy()

        mach = np.empty(shape)
        pressao = np.empty(shape)
        temperatura = np.empty(shape)
        densidade = np.empty(shape)
        velocidade_x = np.empty(shape)
        velocidade_y = np.empty(shape)
        velocidade_z = np.empty(shape)

        for indice, x_local in enumerate(x_grid):
            estado = self._estado_axial(x_local)
            u = estado["VelocidadeAxial"]
            velocidade_radial = (
                u
                * fracao_radial[:, None]
                * self._gradiente_raio(x_local)
            )
            mach[indice, :, :] = estado["Mach"]
            pressao[indice, :, :] = estado["Pressao"]
            temperatura[indice, :, :] = estado["Temperatura"]
            densidade[indice, :, :] = estado["Densidade"]
            velocidade_x[indice, :, :] = u
            velocidade_y[indice, :, :] = velocidade_radial * np.cos(theta)[None, :]
            velocidade_z[indice, :, :] = velocidade_radial * np.sin(theta)[None, :]

        return {
            "x": x,
            "y": y,
            "z": z,
            "Mach": mach,
            "Pressao": pressao,
            "Temperatura": temperatura,
            "Densidade": densidade,
            "u": velocidade_x,
            "v": velocidade_y,
            "w": velocidade_z,
        }

    def exportar_campo_csv(
        self,
        caminho=None,
        n_x=41,
        n_radial=9,
        n_theta=24,
    ):
        """Calcula e salva o campo 3D em CSV, retornando o caminho gerado."""
        campo = self.calcular_campo_3d(n_x, n_radial, n_theta)
        destino = (
            Path(__file__).resolve().parent / "venturi_campo_3d.csv"
            if caminho is None
            else Path(caminho)
        )
        colunas = [
            campo[chave].ravel()
            for chave in (
                "x",
                "y",
                "z",
                "Mach",
                "Pressao",
                "Temperatura",
                "Densidade",
                "u",
                "v",
                "w",
            )
        ]
        np.savetxt(
            destino,
            np.column_stack(colunas),
            delimiter=",",
            header=(
                "x_m,y_m,z_m,Mach,pressao_Pa,temperatura_K,"
                "densidade_kg_m3,u_m_s,v_m_s,w_m_s"
            ),
            comments="",
        )
        return destino


def main():
    simulador = SimulacaoTempoRealVenturi3D()
    simulador.rodar_tempo_real()
    arquivo = simulador.exportar_campo_csv()
    print(f"Campo 3D salvo em: {arquivo}")


if __name__ == "__main__":
    main()
