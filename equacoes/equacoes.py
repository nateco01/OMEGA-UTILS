import math
import numpy as np
from derivadas.derivadas import DerivadasSolver
d = DerivadasSolver()
class EquacoesSolver():
    def __init__(self):
        """
        Solver de equacoes. as listadas sao:
        1. Equacoes do primeiro grau
        2. Equacoes do segundo grau
        """
    
    def EquacaoGrau1(self, a, b, c, d):
        """
        Resolve a*x + b = c*x + d.
        Para resolver a*x + b = 0, passe c = None e d = None (ou c = 0 e d = 0).
        Um c ou um d igual a None é tratado como 0.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        c = 0 if self.c is None else self.c
        d = 0 if self.d is None else self.d

        coef = self.a - c      # coeficiente de x depois de juntar os termos
        termo = d - self.b     # termo independente depois de juntar os termos

        if coef == 0:
            if termo == 0:
                return "A equação possui infinitas soluções reais (0 = 0)."
            return "A equação não possui solução (os termos em x se cancelam e sobra uma igualdade falsa)."

        x = termo / coef
        return f"O resultado para a incógnita é: x = {x}"

    def EquacaoGrau2(self, a, b, c):
        """
        Resolve a*x^2 + b*x + c = 0 (raízes reais) pela fórmula de Bhaskara.
        Se a = 0, a equação deixa de ser do 2º grau e é resolvida como bx + c = 0.
        Para casos onde o discriminante Delta = 0, é aconselhavel que a = 1 
        """
        self.a = a
        self.b = b
        self.c = c
 
        if self.a == 0:
            if self.b == 0:
                if self.c == 0:
                    return "A equação possui infinitas soluções reais (0 = 0)."
                return "A equação não possui solução (a = b = 0 e c != 0)."
            return f"Como a = 0, a equação é do 1º grau: x = {-self.c / self.b}"
 
        delta = (self.b ** 2) - 4 * self.a * self.c
 
        # Com coeficientes inteiros, delta é exato. Com floats, b^2 e 4ac carregam
        # erro de arredondamento: uma raiz dupla como (x - 0.1)^2 pode dar delta
        # levemente positivo ou NEGATIVO. Tolerância RELATIVA à escala dos termos.
        exato = all(isinstance(v, int) for v in (self.a, self.b, self.c))
        if not exato:
            escala = max(self.b ** 2, abs(4 * self.a * self.c))
            if abs(delta) <= 1e-14 * escala:
                delta = 0
 
        if delta > 0:
            raiz = math.sqrt(delta)
            # Evita cancelamento subtrativo: soma b com o MESMO sinal de b e obtém
            # a outra raiz por x1*x2 = c/a (relações de Girard).
            t = -(self.b + math.copysign(raiz, self.b)) / 2
            if self.b >= 0:
                x1, x2 = self.c / t, t / self.a     # x1 = (-b + raiz)/(2a)
            else:
                x1, x2 = t / self.a, self.c / t
            return f"A equação possui duas raízes reais e diferentes: x' = {x1}, x'' = {x2}"
        elif delta == 0:
            x = -self.b / (2 * self.a)
            return f"A equação possui duas raízes reais e iguais: x' = x'' = {x}"
        else:
            return "A equação não possui raízes reais."
        
    def ConversaoLogaritmica(self, x, b):

        self.x = x
        self.b = b

        return  math.log(self.x) / math.log(self.b)

    def DistanciaDoisPontos(self, x1, x2, y1, y2):

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        return math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    def CoeficienteAngular(self, x1, x2, y1, y2):

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def AreaDePoligonos(self, x1, x2, x3, y1, y2, y3):
        """
        Área de um TRIÂNGULO com vértices (x1, y1), (x2, y2) e (x3, y3),
        pela fórmula do determinante. O módulo garante área positiva,
        qualquer que seja a ordem dos vértices.
        """

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3

        return 1/2 * abs(self.x1 * (self.y2 - self.y3) + self.x2 * (self.y3 - self.y1) + self.x3 * (self.y1 - self.y2))

    def Seno(self, x):

        self.x = x

        return math.sin(self.x)

    def Cosseno(self, x):

        self.x = x

        return math.cos(self.x)

    def fatorial(self, n):
        """n! para inteiro n >= 0 (com 0! = 1). Não altera o estado do objeto."""
        if n < 0 or n != int(n):
            raise ValueError("fatorial só é definido para inteiros n >= 0.")

        resultado = 1
        for i in range(2, int(n) + 1):
            resultado *= i
        return resultado
    
    def ProbArranjoSimples(self, n, k):
        """
        Análise Combinatória e Probabilidade

        Arranjo Simples (A ordem importa)
        """

        self.n = n
        self.k = k

        if self.k < 0 or self.k > self.n:
            return "Sinto muito, é preciso 0 <= k <= n."

        return self.fatorial(self.n) // self.fatorial(self.n - self.k)

    def PorbCombinaçãoSimples(self, n, k):
        """
        Análise Combinatória e Probabilidade

        Combinação Simples (A ordem não importa)
        """
        self.n = n
        self.k = k

        if self.k < 0 or self.k > self.n:
            return "Sinto muito, é preciso 0 <= k <= n."

        return self.fatorial(self.n) // (self.fatorial(self.k) * self.fatorial(self.n - self.k))

    def MovimentoUniformeVariado(self, S0, v0, t, a):
        self.S0 = S0
        self.v0 = v0
        self.t = t
        self.a = a

        return self.S0 + self.v0 * self.t + (1/2) * self.a * self.t**2

    def EquacaoDeTorricelli(self, v0, a, S):
        "Voce precisará resolver a equação de movimento uniforme variado (M.U.V) primeiro."
        self.v0 = v0
        self.a = a
        self.S = S
        radicando = v0**2 + 2 * self.a * S
        if radicando < 0:
            raise ValueError("Os dados produzem velocidade final nao real.")
        return math.sqrt(radicando)

    def SegundaLeiNewton(self, m, a):
        """
        Segunda Lei de Newton: F = m * a
        Calcula a força resultante.
        """
        self.m = m
        self.a = a
        if m < 0:
            raise ValueError("A massa não pode ser negativa.")
        return self.m * self.a 

    def EnergiaCinetica(self, m, v):
        self.m = m
        self.v = v

        return (1/2) * self.m * self.v**2

    def PotencialGravitacional(self, m, g, h):
        self.m = m
        self.g = g
        self.h = h

        return self.m * self.g * self.h

    def Densidade(self, m, V):
        """
        A densidade de um corpo ou objeto pode ser determinada
        pela quantidade de massa que o mesmo possui, dividio
        pelo volume que este ocupa.
        """
        if not math.isfinite(float(m)) or m < 0:
            raise ValueError("A massa deve ser finita e nao negativa.")
        if not math.isfinite(float(V)) or V <= 0:
            raise ValueError("O volume deve ser positivo e finito.")
        return m / V

    def DensidadeRelativa(self, densidade, densidade_referencia):
        """
        Razao adimensional entre densidade da amostra e densidade de referencia.
        """
        densidade = float(densidade)
        densidade_referencia = float(densidade_referencia)
        if not math.isfinite(densidade) or densidade < 0.0:
            raise ValueError("A densidade deve ser finita e nao negativa.")
        if not math.isfinite(densidade_referencia) or densidade_referencia <= 0.0:
            raise ValueError("A densidade de referencia deve ser positiva e finita.")
        return densidade / densidade_referencia

    def AceleracaoArrastoMagnetico(self, sigma, B, rho, v):
        """
        Calcula a aceleração de frenagem devido à força de Lorentz em um fluido condutor (Magnetohidrodinâmica).
        
        Parâmetros:
            sigma : Condutividade elétrica do fluido (S/m)
            B     : Intensidade do campo magnético (Tesla)
            rho   : Densidade do fluido (kg/m³)
            v     : Velocidade instantânea do fluido (m/s)
            
        Retorna a aceleração (negativa, pois é uma força de arrasto).
        """
        valores = (sigma, B, rho, v)
        if not all(math.isfinite(float(valor)) for valor in valores):
            raise ValueError("Os parametros do arrasto magnetico devem ser finitos.")
        if sigma < 0 or B < 0 or rho <= 0:
            raise ValueError("sigma e B devem ser nao negativos e rho positivo.")
        return -((sigma * (B ** 2)) / rho) * v

    def AceleracaoArrastoAerodinamico(self, cd, area, massa, rho, v):
        """
        Calcula a aceleração de frenagem gerada pelo atrito de um objeto contra um fluido (como o ar).
        
        Parâmetros:
            cd    : Coeficiente de arrasto aerodinâmico
            area  : Área de seção transversal (m²)
            massa : Massa do objeto (kg)
            rho   : Densidade do fluido/ar no momento (kg/m³)
            v     : Velocidade instantânea (m/s)
        """
        valores = (cd, area, massa, rho, v)
        if not all(math.isfinite(float(valor)) for valor in valores):
            raise ValueError("Os parametros do arrasto aerodinamico devem ser finitos.")
        if cd < 0 or area < 0 or rho < 0 or massa <= 0:
            raise ValueError("cd, area e rho devem ser nao negativos; massa deve ser positiva.")
        return -(0.5 * cd * area * rho * abs(v) * v) / massa

    def DensidadeBarometrica(self, rho0, y, H_escala):
        """
        Calcula a densidade atmosférica em uma determinada altitude (y), assumindo decaimento exponencial.
        
        Parâmetros:
            rho0     : Densidade do ar ao nível do mar (ex: 1.225 kg/m³)
            y        : Altitude atual (m)
            H_escala : Altura de escala da atmosfera (ex: 8500 m para a Terra)
        """
        if not all(math.isfinite(float(valor)) for valor in (rho0, y, H_escala)):
            raise ValueError("rho0, y e H_escala devem ser finitos.")
        if rho0 < 0 or H_escala <= 0:
            raise ValueError("rho0 deve ser nao negativa e H_escala positiva.")
        return rho0 * math.exp(-y / H_escala)

    def CalorGeradoPorFriccao(self, massa, v, eficiencia):
        """
        Converte a energia cinética de um impacto ou arrasto contínuo em energia térmica (Calor).
        
        Parâmetros:
            massa      : Massa do objeto (kg)
            v          : Velocidade dissipada (m/s)
            eficiencia : Fração da energia que se transforma efetivamente em calor no objeto (0.0 a 1.0)
        """
        if not all(math.isfinite(float(valor)) for valor in (massa, v, eficiencia)):
            raise ValueError("massa, v e eficiencia devem ser finitos.")
        if massa < 0 or not 0.0 <= eficiencia <= 1.0:
            raise ValueError("massa deve ser nao negativa e eficiencia deve estar entre 0 e 1.")
        return 0.5 * massa * (v ** 2) * eficiencia

    def ForcaDeLorentz(self, q, E, B, v):
        """Forca de Lorentz vetorial F = q (E + v x B), em SI."""
        carga = float(q)
        campo_eletrico = np.asarray(E, dtype=float)
        campo_magnetico = np.asarray(B, dtype=float)
        velocidade = np.asarray(v, dtype=float)
        if not math.isfinite(carga):
            raise ValueError("A carga deve ser finita.")
        if any(
            vetor.shape != (3,) or not np.all(np.isfinite(vetor))
            for vetor in (campo_eletrico, campo_magnetico, velocidade)
        ):
            raise ValueError("E, B e v devem ser vetores tridimensionais finitos.")
        return carga * (campo_eletrico + np.cross(velocidade, campo_magnetico))

    def EquacaoDeKepler(self, e, M, n):
        """
        Equação de Kepler metodo ponto fixo para orbitas elipticas.\n
        Tambem pode ser calculada para orbitas circunferecncial alterando o parametro e.
        """
        E = M
        lista = []
        for i in range(n):
            k = M + e * math.sin(E)
            E = k
            lista.append(k)

        return lista

    def EquacaoDeKeplerRadial(self, x, max_inter, tol, sinal="+"):
        """
        Equação de Kepler para trajetoria radial.
        """
        t = x
        lista = []
        for i in range(max_inter):
            if sinal == "+":
                k = ((math.asinh(math.sqrt(t))) - math.sqrt(t*(1+t)))
                t = k
                lista.append(t)
                if t <= tol:
                    break
            if sinal == "-":
                k = -((math.asinh(math.sqrt(t))) - math.sqrt(t*(1+t)))
                t = k
                lista.append(t)
                if t <= tol:
                    break
        return lista

    def EquacaoDeKeplerBisessao(self, M, e, tol, max_inter):
        from funcoes import FuncoesSolverNaoLinear
        f = FuncoesSolverNaoLinear()

        a = M - e if M - e > 0 else 0.0
        b = M + e if M + e < 2 * 3.1415926535 else 2 * 3.1415926535

        func = lambda E: E - e * math.sin(e) - M

        return f.MetodoBissecao(func, a, b, max_inter, tol)

class EDOsSolver():
    def __init__(self):
        pass

    def Rk4(self, f, y0, h, passos, t0=0.0):
        """
        Método de Runge-Kutta de 4ª Ordem (RK4) para EDOs dy/dt = f(t, y).

        f      : callable f(t, y) que retorna a derivada dy/dt
        y0     : valor inicial de y em t0
        h      : tamanho do passo
        passos : número de passos da integração
        t0     : tempo inicial (default: 0.0)

        Devolve (ts, ys): a lista dos instantes e a lista dos valores de y em
        cada instante, ambas com passos + 1 elementos (o primeiro é o estado
        inicial). O valor final é ys[-1], no instante ts[-1].
        """
        y = y0
        t = t0
        ts = [t]
        ys = [y]

        for _ in range(passos):
            k1 = h * f(t, y)
            k2 = h * f(t + h / 2.0, y + k1 / 2.0)
            k3 = h * f(t + h / 2.0, y + k2 / 2.0)
            k4 = h * f(t + h, y + k3)

            y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            t = t + h

            ts.append(t)
            ys.append(y)

        return ts, ys

    def Rk45(
        self,
        f,
        y0,
        t0,
        t_final,
        passo_inicial=None,
        tolerancia_absoluta=1e-9,
        tolerancia_relativa=1e-7,
        max_passos=100000,
    ):
        """Integra EDOs com Dormand-Prince 5(4) e controle local de erro.

        f(t, y) deve retornar derivadas com o mesmo formato de y. Retorna
        (tempos, estados, diagnosticos); os estados sao floats para escalar e
        arrays NumPy para estado vetorial. O erro e controlado localmente, nao
        e uma prova de erro global limitado.
        """
        if not callable(f):
            raise TypeError("f deve ser uma funcao de derivada f(t, y).")
        t = float(t0)
        t_final = float(t_final)
        atol = float(tolerancia_absoluta)
        rtol = float(tolerancia_relativa)
        if not all(math.isfinite(valor) for valor in (t, t_final, atol, rtol)):
            raise ValueError("Tempos e tolerancias devem ser finitos.")
        if atol <= 0.0 or rtol < 0.0 or (atol == 0.0 and rtol == 0.0):
            raise ValueError("Use tolerancia_absoluta > 0 e tolerancia_relativa >= 0.")
        if isinstance(max_passos, bool) or not isinstance(max_passos, int) or max_passos < 1:
            raise ValueError("max_passos deve ser um inteiro positivo.")

        y = np.asarray(y0, dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError("y0 deve conter valores finitos.")
        escalar = y.ndim == 0
        direcao = 1.0 if t_final >= t else -1.0
        intervalo = abs(t_final - t)
        if intervalo == 0.0:
            estado_inicial = float(y) if escalar else y.copy()
            return [t], [estado_inicial], {
                "passos_aceitos": 0,
                "passos_rejeitados": 0,
                "erro_normalizado_final": 0.0,
            }

        passo_abs = (
            intervalo / 100.0
            if passo_inicial is None
            else abs(float(passo_inicial))
        )
        if not math.isfinite(passo_abs) or passo_abs <= 0.0:
            raise ValueError("passo_inicial deve ser positivo e finito.")
        h = direcao * min(passo_abs, intervalo)
        tempos = [t]
        estados = [float(y) if escalar else y.copy()]
        aceitos = rejeitados = 0
        erro_final = 0.0

        def derivada(tempo, estado):
            entrada = float(estado) if escalar else estado.copy()
            valor = np.asarray(f(tempo, entrada), dtype=float)
            if valor.shape != y.shape or not np.all(np.isfinite(valor)):
                raise ValueError("f(t, y) deve retornar derivadas finitas com o shape de y.")
            return valor

        while direcao * (t_final - t) > 0.0:
            if aceitos + rejeitados >= max_passos:
                raise RuntimeError("Rk45 excedeu max_passos antes de atingir t_final.")
            if direcao * (t + h - t_final) > 0.0:
                h = t_final - t

            k1 = derivada(t, y)
            k2 = derivada(t + h / 5.0, y + h * (k1 / 5.0))
            k3 = derivada(
                t + 3.0 * h / 10.0,
                y + h * (3.0 * k1 / 40.0 + 9.0 * k2 / 40.0),
            )
            k4 = derivada(
                t + 4.0 * h / 5.0,
                y
                + h
                * (
                    44.0 * k1 / 45.0
                    - 56.0 * k2 / 15.0
                    + 32.0 * k3 / 9.0
                ),
            )
            k5 = derivada(
                t + 8.0 * h / 9.0,
                y
                + h
                * (
                    19372.0 * k1 / 6561.0
                    - 25360.0 * k2 / 2187.0
                    + 64448.0 * k3 / 6561.0
                    - 212.0 * k4 / 729.0
                ),
            )
            k6 = derivada(
                t + h,
                y
                + h
                * (
                    9017.0 * k1 / 3168.0
                    - 355.0 * k2 / 33.0
                    + 46732.0 * k3 / 5247.0
                    + 49.0 * k4 / 176.0
                    - 5103.0 * k5 / 18656.0
                ),
            )
            y5 = y + h * (
                35.0 * k1 / 384.0
                + 500.0 * k3 / 1113.0
                + 125.0 * k4 / 192.0
                - 2187.0 * k5 / 6784.0
                + 11.0 * k6 / 84.0
            )
            k7 = derivada(t + h, y5)
            y4 = y + h * (
                5179.0 * k1 / 57600.0
                + 7571.0 * k3 / 16695.0
                + 393.0 * k4 / 640.0
                - 92097.0 * k5 / 339200.0
                + 187.0 * k6 / 2100.0
                + k7 / 40.0
            )

            escala = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
            erro_final = float(np.sqrt(np.mean(((y5 - y4) / escala) ** 2)))
            if not math.isfinite(erro_final):
                erro_final = math.inf

            if erro_final <= 1.0:
                t += h
                y = y5
                tempos.append(t)
                estados.append(float(y) if escalar else y.copy())
                aceitos += 1
            else:
                rejeitados += 1

            fator = 5.0 if erro_final == 0.0 else min(5.0, max(0.2, 0.9 * erro_final ** (-0.2)))
            h *= fator
            passo_minimo = np.finfo(float).eps * max(1.0, abs(t), abs(t_final))
            if abs(h) < passo_minimo and direcao * (t_final - t) > 0.0:
                raise RuntimeError("Rk45 atingiu o passo minimo antes de convergir.")

        return tempos, estados, {
            "passos_aceitos": aceitos,
            "passos_rejeitados": rejeitados,
            "erro_normalizado_final": erro_final,
        }

    def EdoPrimeiraOrdem(self, f):
        """
        As equações diferenciais ordfinarias de primeira ordem
        são da forma F(x, y, y') = 0
        Mas geralmente por meio de simples manipulação algebrica
        conseguem-se reescrever na forma de uma ou mais equações.
        dy/dx = f(x, y)
        """
        return f

    def EdoPrimeiraOrdemInversa(self, f):
        """
        Forma inversa da EDO de primeira ordem. Se dy/dx = f(x, y), então,
        onde f != 0:

            dx/dy = 1 / f(x, y)

        Devolve a função g(y, x) = 1 / f(x, y), no formato que o Rk4 espera
        (variável independente primeiro). Para integrar x como função de y,
        partindo de x = x0 em y = y0:

            g = q.EdoPrimeiraOrdemInversa(f)
            ys, xs = q.Rk4(g, x0, h, passos, t0=y0)   # aqui "ts" são os valores de y

        Útil quando y é a melhor variável independente (por exemplo, perto de
        pontos onde a tangente da solução é vertical, f -> infinito).
        """
        def g(y, x):
            fxy = f(x, y)
            if fxy == 0:
                raise ZeroDivisionError("f(x, y) = 0: a forma inversa não está definida neste ponto.")
            return 1.0 / fxy

        return g

    def EdoMetodoSeparacaoVariaveis(self, funcao_x, funcao_y):
        """Monta o campo dy/dx=f(x)g(y) para integrar com Rk4 ou Rk45."""
        if not callable(funcao_x) or not callable(funcao_y):
            raise TypeError("funcao_x e funcao_y devem ser funcoes numericas.")

        def derivada(x, y):
            return funcao_x(x) * funcao_y(y)

        return derivada
    
    def EquacaoDoCalor1D(self, x, t, k, c, rho):
        """
        Função de Green da equação do calor 1D, u_t = sigma * u_xx:

            G(x, t) = exp(-x² / (4*sigma*t)) / sqrt(4*pi*sigma*t),   t > 0

        onde sigma = k / (c * rho) é a difusividade térmica (m²/s).
        Parametros:
            x   : posição (m) em relação à fonte
            t   : tempo (s) desde que a fonte instantânea foi aplicada (t > 0)
            k   : condutividade térmica (W/(m·K))
            c   : calor específico (J/(kg·K))
            rho : densidade (kg/m³)

        G tem unidade 1/m e integra 1 em x (conservação de energia). Para uma
        fonte plana instantânea de energia Q por unidade de área em x = 0, o
        aumento de temperatura é  dT(x, t) = Q / (c * rho) * G(x, t).

        Esta função pode ser usada em um loop (em t ou em x) para visualizar a evolução.
        """
        if not all(math.isfinite(float(valor)) for valor in (x, t, k, c, rho)):
            return "Sinto muito, x, t, k, c e rho devem ser finitos."
        if t <= 0:
            return "Sinto muito, t deve ser positivo (em t = 0 a solução é uma delta de Dirac)."
        if k <= 0 or c <= 0 or rho <= 0:
            return "Sinto muito, k, c e rho devem ser positivos."

        sigma = k / (c * rho)

        return (1 / math.sqrt(4 * math.pi * sigma * t)) * math.exp(-(x ** 2) / (4 * sigma * t))

    def EquacaoDoCalor3D(self, x, y, z, t, k, c, rho):
        """
        Função de Green da equação do calor 3D, u_t = sigma * (u_xx + u_yy + u_zz):

            G(x, y, z, t) = exp(-(x² + y² + z²) / (4*sigma*t)) / (4*pi*sigma*t)^(3/2),   t > 0

        onde sigma = k / (c * rho) é a difusividade térmica (m²/s). Vale para um
        meio infinito, homogêneo e isotrópico, sem fontes.

        Como o problema é separável, G é o produto das três funções de Green 1D,
        G = G1(x, t) * G1(y, t) * G1(z, t), que é como esta função é calculada
        (reaproveita EquacaoDoCalor1D).

        Parametros:
            x, y, z : posição (m) em relação à fonte
            t       : tempo (s) desde que a fonte instantânea foi aplicada (t > 0)
            k       : condutividade térmica (W/(m·K))
            c       : calor específico (J/(kg·K))
            rho     : densidade (kg/m³)

        G tem unidade 1/m³ e integra 1 sobre todo o espaço. Para uma fonte pontual
        instantânea de energia Q em (0, 0, 0), o aumento de temperatura é
        dT(x, y, z, t) = Q / (c * rho) * G(x, y, z, t).
        """
        gx = self.EquacaoDoCalor1D(x, t, k, c, rho)
        gy = self.EquacaoDoCalor1D(y, t, k, c, rho)
        gz = self.EquacaoDoCalor1D(z, t, k, c, rho)
        for resultado in (gx, gy, gz):
            if isinstance(resultado, str):
                return resultado

        return gx * gy * gz

    def MovimentoHarmonico(self, A, B, w, t, k, m):
        if w is None:
            if m <= 0 or k < 0:
                raise ValueError("Para calcular w=sqrt(k/m), use k >= 0 e m > 0.")
            w = math.sqrt(k / m)
        if not math.isfinite(float(w)):
            raise ValueError("w deve ser finito.")
        return (A * math.cos(w * t)) + (B * math.sin(w * t))

    def EquacaoDaOnda1D(self, c, f, x=None, y=None, z=None):
        """
        Equação da Onda em 1D. para u utiliza-se o laplaciano
        disponivel no pacote de derivadas do OMEGA.
        c é uma constante
        """
        from derivadas.derivadas import DerivadasSolver
        d = DerivadasSolver()
        return (c**2) * d.Laplaciano(f, x, y, z)

    
    #Soluções analíticas exatas para escoamentos clássicos de fluidos (Navier-Stokes).
    

    def EscoamentoPoiseuilleTubo(self, r, R, mu, dp_dx):
        """
        u(r) = 1/(4*mu) * (-dp/dx) * (R^2 - r^2)
        r  : raio atual (0 <= r <= R)
        R  : raio do tubo
        mu : viscosidade dinâmica
        dp_dx : gradiente de pressão (dp/dx)
        """
        if not all(math.isfinite(float(valor)) for valor in (r, R, mu, dp_dx)):
            raise ValueError("r, R, mu e dp_dx devem ser finitos.")
        if R <= 0.0 or mu <= 0.0:
            raise ValueError("R e mu devem ser positivos.")
        if not 0.0 <= r <= R:
            raise ValueError("r deve estar entre 0 e R.")
        return (1.0 / (4.0 * mu)) * (-dp_dx) * (R**2 - r**2)

    def EscoamentoPoiseuillePlano(self, y, h, mu, dp_dx):
        """
        u(y) = 1/(2*mu) * (-dp/dx) * y * (h - y)
        y  : posição transversal (0 <= y <= h)
        h  : separação entre as placas
        """
        if not all(math.isfinite(float(valor)) for valor in (y, h, mu, dp_dx)):
            raise ValueError("y, h, mu e dp_dx devem ser finitos.")
        if h <= 0.0 or mu <= 0.0:
            raise ValueError("h e mu devem ser positivos.")
        if not (0 <= y <= h):
            raise ValueError("A coordenada y deve estar entre 0 e h.")
        return (1.0 / (2.0 * mu)) * (-dp_dx) * y * (h - y)

    def EscoamentoCouetteSimples(self, y, h, U):
        """
        u(y) = U * (y / h)
        U  : velocidade da placa móvel
        """
        if not all(math.isfinite(float(valor)) for valor in (y, h, U)):
            raise ValueError("y, h e U devem ser finitos.")
        if h <= 0.0 or not 0.0 <= y <= h:
            raise ValueError("h deve ser positivo e y deve estar entre 0 e h.")
        return U * (y / h)

    def EscoamentoCouettePoiseuille(self, y, h, U, mu, dp_dx):
        """
        u(y) = U*(y/h) + 1/(2*mu)*(-dp/dx)*y*(h-y)
        """
        termo_couette = self.EscoamentoCouetteSimples(y, h, U)
        termo_poiseuille = self.EscoamentoPoiseuillePlano(y, h, mu, dp_dx)
        return termo_couette + termo_poiseuille

    def PrimeiroProblemaStokes(self, y, t, U, nu):
        """
        u(y, t) = U * erfc(y / (2 * sqrt(nu * t)))
        nu : viscosidade cinemática ( kinematic viscosity )
        t  : tempo (> 0)
        """
        if not all(math.isfinite(float(valor)) for valor in (y, t, U, nu)):
            raise ValueError("y, t, U e nu devem ser finitos.")
        if nu <= 0.0:
            raise ValueError("nu deve ser positiva.")
        if t <= 0:
            return U if y == 0 else 0.0
        
        arg = y / (2.0 * math.sqrt(nu * t))
        return U * math.erfc(arg)

    def SegundoProblemaStokes(self, y, t, U, omega, nu):
        """
        u(y, t) = U * exp(-y / delta_s) * cos(omega * t - y / delta_s)
        omega : frequência angular da oscilação
        """
        if not all(math.isfinite(float(valor)) for valor in (y, t, U, omega, nu)):
            raise ValueError("y, t, U, omega e nu devem ser finitos.")
        if omega <= 0.0 or nu <= 0.0:
            raise ValueError("omega e nu devem ser positivos.")
        delta_s = math.sqrt((2.0 * nu) / omega)
        return U * math.exp(-y / delta_s) * math.cos(omega * t - (y / delta_s))

    def NavierStokesConvectiva3D(self, func_u, func_v, func_w, func_p, x, y, z, rho, nu, fx=0.0, fy=0.0, fz=0.0):
        """
        Calcula as derivadas temporais (aceleração local) do fluido em 3D 
        pela equação de Navier-Stokes incompressível na forma convectiva:
            ∂u/∂t = -(u·∇)u - (1/rho)∇p + nu*∇²u + f

        Parâmetros:
            func_u, func_v, func_w : funções (callables) que retornam as componentes da velocidade.
            func_p                 : função (callable) que retorna a pressão.
            x, y, z                : coordenadas espaciais (DEVEM ser instâncias de tf.Variable).
            rho                    : densidade do fluido.
            nu                     : viscosidade cinemática do fluido.
            fx, fy, fz             : forças externas (como gravidade). Default = 0.

        Retorno:
            du_dt, dv_dt, dw_dt    : tensores representando as acelerações em x, y e z.
        """
        import tensorflow as tf

        # Verificação de segurança: x, y, z precisam ser variáveis rastreáveis do TensorFlow
        if not all(isinstance(var, tf.Variable) for var in [x, y, z]):
            raise TypeError("As coordenadas x, y e z precisam ser instâncias de tf.Variable para o cálculo do gradiente.")
        if not (x.dtype == y.dtype == z.dtype):
            raise TypeError("As coordenadas x, y e z devem ter o mesmo dtype.")
        if not math.isfinite(float(rho)) or rho <= 0.0:
            raise ValueError("rho deve ser positiva e finita.")
        if not math.isfinite(float(nu)) or nu < 0.0:
            raise ValueError("nu deve ser nao negativa e finita.")
        rho = tf.cast(rho, x.dtype)
        nu = tf.cast(nu, x.dtype)
        fx, fy, fz = (tf.cast(tf.convert_to_tensor(valor), x.dtype) for valor in (fx, fy, fz))

        def gradientes_seguros(tape, alvo, fontes):
            gradientes = tape.gradient(alvo, fontes)
            return tuple(
                tf.zeros_like(fonte) if gradiente is None else gradiente
                for gradiente, fonte in zip(gradientes, fontes)
            )

        def campo(funcao):
            valor = tf.cast(tf.convert_to_tensor(funcao(x, y, z)), x.dtype)
            if valor.shape.rank != 0:
                raise ValueError("Cada componente do campo deve retornar um escalar.")
            return valor

        # Utilizamos dois 'tapes' aninhados: 
        # tape1 calcula a 1ª derivada (gradientes / convecção)
        # tape2 calcula a 2ª derivada em cima da 1ª (laplaciano / difusão)
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                
                # Avalia os campos de velocidade e pressão no espaço (x, y, z)
                u = campo(func_u)
                v = campo(func_v)
                w = campo(func_w)
                p = campo(func_p)

            # --- 1. CÁLCULO DOS GRADIENTES (1ª Derivada) ---
            # Gradientes de velocidade
            du_dx, du_dy, du_dz = gradientes_seguros(tape1, u, [x, y, z])
            dv_dx, dv_dy, dv_dz = gradientes_seguros(tape1, v, [x, y, z])
            dw_dx, dw_dy, dw_dz = gradientes_seguros(tape1, w, [x, y, z])
            
            # Gradientes de pressão
            dp_dx, dp_dy, dp_dz = gradientes_seguros(tape1, p, [x, y, z])

        # --- 2. CÁLCULO DOS LAPLACIANOS (2ª Derivada) ---
        d2u_dx2 = gradientes_seguros(tape2, du_dx, [x])[0]
        d2u_dy2 = gradientes_seguros(tape2, du_dy, [y])[0]
        d2u_dz2 = gradientes_seguros(tape2, du_dz, [z])[0]
        laplaciano_u = d2u_dx2 + d2u_dy2 + d2u_dz2

        d2v_dx2 = gradientes_seguros(tape2, dv_dx, [x])[0]
        d2v_dy2 = gradientes_seguros(tape2, dv_dy, [y])[0]
        d2v_dz2 = gradientes_seguros(tape2, dv_dz, [z])[0]
        laplaciano_v = d2v_dx2 + d2v_dy2 + d2v_dz2

        d2w_dx2 = gradientes_seguros(tape2, dw_dx, [x])[0]
        d2w_dy2 = gradientes_seguros(tape2, dw_dy, [y])[0]
        d2w_dz2 = gradientes_seguros(tape2, dw_dz, [z])[0]
        laplaciano_w = d2w_dx2 + d2w_dy2 + d2w_dz2

        # Limpamos a memória dos gravadores, pois não precisamos mais derivar
        del tape1
        del tape2

        # --- 3. TERMOS CONVECTIVOS: (u·∇)u ---
        convectivo_u = u * du_dx + v * du_dy + w * du_dz
        convectivo_v = u * dv_dx + v * dv_dy + w * dv_dz
        convectivo_w = u * dw_dx + v * dw_dy + w * dw_dz

        # --- 4. MONTAGEM DA EQUAÇÃO DE NAVIER-STOKES ---
        # du/dt = - Convecção - Gradiente de Pressão + Difusão + Forças Externas
        du_dt = -convectivo_u - (1.0 / rho) * dp_dx + nu * laplaciano_u + fx
        dv_dt = -convectivo_v - (1.0 / rho) * dp_dy + nu * laplaciano_v + fy
        dw_dt = -convectivo_w - (1.0 / rho) * dp_dz + nu * laplaciano_w + fz

        return du_dt, dv_dt, dw_dt
    
    
    def NavierStokesConservativa3D(self, func_u, func_v, func_w, func_p, x, y, z, rho, nu, fx=0.0, fy=0.0, fz=0.0):
        """
        Calcula as derivadas temporais (aceleração local) do fluido em 3D 
        pela equação de Navier-Stokes incompressível na forma conservativa:
            ∂u/∂t = -∇·(u ⊗ u) - (1/rho)∇p + nu*∇²u + f

        Parâmetros:
            func_u, func_v, func_w : funções (callables) que retornam as componentes da velocidade.
            func_p                 : função (callable) que retorna a pressão.
            x, y, z                : coordenadas espaciais (instâncias de tf.Variable).
            rho                    : densidade do fluido.
            nu                     : viscosidade cinemática do fluido.
            fx, fy, fz             : forças externas. Default = 0.

        Retorno:
            du_dt, dv_dt, dw_dt    : tensores representando as acelerações em x, y e z.
        """
        import tensorflow as tf

        if not all(isinstance(var, tf.Variable) for var in [x, y, z]):
            raise TypeError("As coordenadas x, y e z precisam ser instâncias de tf.Variable.")
        if not (x.dtype == y.dtype == z.dtype):
            raise TypeError("As coordenadas x, y e z devem ter o mesmo dtype.")
        if not math.isfinite(float(rho)) or rho <= 0.0:
            raise ValueError("rho deve ser positiva e finita.")
        if not math.isfinite(float(nu)) or nu < 0.0:
            raise ValueError("nu deve ser nao negativa e finita.")
        rho = tf.cast(rho, x.dtype)
        nu = tf.cast(nu, x.dtype)
        fx, fy, fz = (tf.cast(tf.convert_to_tensor(valor), x.dtype) for valor in (fx, fy, fz))

        fontes = [x, y, z]

        def gradientes_seguros(tape, alvo, variaveis):
            gradientes = tape.gradient(alvo, variaveis)
            return tuple(
                tf.zeros_like(variavel) if gradiente is None else gradiente
                for gradiente, variavel in zip(gradientes, variaveis)
            )

        def campo(funcao):
            valor = tf.cast(tf.convert_to_tensor(funcao(x, y, z)), x.dtype)
            if valor.shape.rank != 0:
                raise ValueError("Cada componente do campo deve retornar um escalar.")
            return valor

        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                
                u = campo(func_u)
                v = campo(func_v)
                w = campo(func_w)
                p = campo(func_p)

                # Componentes do tensor de fluxo de momento (u ⊗ u)
                uu = u * u
                uv = u * v
                uw = u * w
                vv = v * v
                vw = v * w
                ww = w * w

            # --- 1. CÁLCULO DOS GRADIENTES (1ª Derivada) ---
            
            # Gradientes de pressão
            dp_dx, dp_dy, dp_dz = gradientes_seguros(tape1, p, fontes)

            # Divergente do fluxo de momento para a componente X: ∂(uu)/∂x + ∂(uv)/∂y + ∂(uw)/∂z
            duu_dx = gradientes_seguros(tape1, uu, [x])[0]
            duv_dy = gradientes_seguros(tape1, uv, [y])[0]
            duw_dz = gradientes_seguros(tape1, uw, [z])[0]
            
            # Divergente do fluxo de momento para a componente Y: ∂(uv)/∂x + ∂(vv)/∂y + ∂(vw)/∂z
            # (uv = vu, então usamos uv)
            duv_dx = gradientes_seguros(tape1, uv, [x])[0]
            dvv_dy = gradientes_seguros(tape1, vv, [y])[0]
            dvw_dz = gradientes_seguros(tape1, vw, [z])[0]

            # Divergente do fluxo de momento para a componente Z: ∂(uw)/∂x + ∂(vw)/∂y + ∂(ww)/∂z
            duw_dx = gradientes_seguros(tape1, uw, [x])[0]
            dvw_dy = gradientes_seguros(tape1, vw, [y])[0]
            dww_dz = gradientes_seguros(tape1, ww, [z])[0]

            # Para o termo difusivo (laplaciano), ainda precisamos dos gradientes simples da velocidade
            du_dx, du_dy, du_dz = gradientes_seguros(tape1, u, fontes)
            dv_dx, dv_dy, dv_dz = gradientes_seguros(tape1, v, fontes)
            dw_dx, dw_dy, dw_dz = gradientes_seguros(tape1, w, fontes)

        # --- 2. CÁLCULO DOS LAPLACIANOS (2ª Derivada) ---
        d2u_dx2 = gradientes_seguros(tape2, du_dx, [x])[0]
        d2u_dy2 = gradientes_seguros(tape2, du_dy, [y])[0]
        d2u_dz2 = gradientes_seguros(tape2, du_dz, [z])[0]
        laplaciano_u = d2u_dx2 + d2u_dy2 + d2u_dz2

        d2v_dx2 = gradientes_seguros(tape2, dv_dx, [x])[0]
        d2v_dy2 = gradientes_seguros(tape2, dv_dy, [y])[0]
        d2v_dz2 = gradientes_seguros(tape2, dv_dz, [z])[0]
        laplaciano_v = d2v_dx2 + d2v_dy2 + d2v_dz2

        d2w_dx2 = gradientes_seguros(tape2, dw_dx, [x])[0]
        d2w_dy2 = gradientes_seguros(tape2, dw_dy, [y])[0]
        d2w_dz2 = gradientes_seguros(tape2, dw_dz, [z])[0]
        laplaciano_w = d2w_dx2 + d2w_dy2 + d2w_dz2

        del tape1
        del tape2

        # --- 3. TERMOS CONSERVATIVOS ---
        div_fluxo_u = duu_dx + duv_dy + duw_dz
        div_fluxo_v = duv_dx + dvv_dy + dvw_dz
        div_fluxo_w = duw_dx + dvw_dy + dww_dz

        # --- 4. MONTAGEM DA EQUAÇÃO ---
        du_dt = -div_fluxo_u - (1.0 / rho) * dp_dx + nu * laplaciano_u + fx
        dv_dt = -div_fluxo_v - (1.0 / rho) * dp_dy + nu * laplaciano_v + fy
        dw_dt = -div_fluxo_w - (1.0 / rho) * dp_dz + nu * laplaciano_w + fz

        return du_dt, dv_dt, dw_dt

    def TaxaDeDecaimento(self, M, p, max_inter, modo="puro"):
        """
        Função para calcular taxa de decaimento de um elemento.\n
        Parametros:\n
        M: é a massa do elemento, podendo ser inserida como unidade atomica, ou kg.\n
        p: é a constante de decaimento, fique a vontede para conferir o modulo de constantes do Omega para casos mais especializados.\n
        max_inter: numero de interações para verificar a evolucao.\n
        modo: você pode escolher entre puro (retornar os dados em uma lista de dicionarios) e estruct (retorna dados simplificados para facil leitura)
        """
        lista = []
        for i in range(max_inter):
            x = M * math.exp(-p * i)
            dic = {"M": x, "Instante": i}
            lista.append(dic)

        if modo == "puro":
            return lista

        if modo == "estruct":
            for item in lista:
                print(f"Instante {item['Instante']}: M = {item['M']:.4f}")

            # Movido para FORA do loop for:
            massa_final = lista[-1]["M"]
            porcentagem_restante = (massa_final / M) * 100
            print(f"\nMassa restante: {porcentagem_restante:.8f}%")

    def SecaoDeChoqueRutherford(self, Z, e, epslon, E, theta):
        """Espalhamento columbiano simples entre um eletron de carga negativa e um nucleo de carga positiva.

        Sem considerar o efeito de blindagem da nuvem eletronica.\n
        Parametros:
        Z: Número atomico do material.
        e: Carga elementar do eletron (disponivel no modulo de constantes do omega)
        epslon: permissividade do vacuo (disponivel no modulo de constantes do omega)
        E: Energia cinetica do eletron
        theta: O angulo, pode ser por exemplo: theta = math.pi / 3  # Exemplo: 60 graus
        """
        if theta == 0:
            return None

        return (
            ((Z * e**2) / (16 * math.pi * epslon * E)) ** 2
        ) * 1 / (math.sin(theta / 2) ** 4)

    def SecaoDeChoqueRutherfordBlindada(self, Z, e, epslon, E, theta, sigma=0.0, mode_sigma="auto"):
        """Espalhamento columbiano simples com blindagem da nuvem eletronica.\n

        Parametros:
        Z: Número atomico do material.
        e: Carga elementar do eletron
        epslon: permissividade do vacuo
        E: Energia cinetica do eletron
        theta: O angulo
        sigma: parametro de screening
        mode_sigma: "auto" ou "manual"
        """
        if mode_sigma == "auto":
        # Corrigido math.exp(0.67) para Z**0.67 conforme padrão físico de screening
            sigma = 5.5e-4 * (Z**0.67 / E)

        denominador = (4 * math.sin(theta / 2) ** 2 + sigma) ** 2
        if denominador == 0:
            return 0.0

        return (((Z * e**2) / (16 * math.pi * epslon * E)) ** 2) * (
            1 / denominador
        )
        

    def SecaoChoqueMott(self, Z, E_kev, theta, alpha, e, eps0, m_e, c):  # Ajustado a ordem dos parâmetros para evitar erros de sintaxe
        """Calcula a seção de choque diferencial de Mott com correção relativística

        e parâmetro de blindagem (screening) opcional.
        Baseado nas equações da dissertação (Capítulo 4 e 5).
        """
        # Energia em Joules (Corrigido o operador de multiplicação)
        E_joules = E_kev * 1000 * e

        # Velocidade v e fator relativístico beta (v/c) (Corrigido o operador)
        v = math.sqrt(2 * E_joules / m_e)
        beta = v / c

        # Termo base de Rutherford (Corrigido os operadores de multiplicação faltantes)
        sin_half_theta = math.sin(theta / 2.0)
        termo_rutherford = (
            (Z * e**2) / (4 * math.pi * eps0 * (4 * E_joules))
        ) ** 2

        # Seção de choque com blindagem e correção de Mott (Corrigido operadores de multiplicação)
        denominador = (sin_half_theta**2 + alpha**2) ** 2
        if denominador == 0:
            return 0.0

        d_sigma_omega = (
            termo_rutherford
            / denominador
            * (1 - beta**2 * (sin_half_theta**2))
            * 10000
        )  # Conversão para cm² se necessário
        return d_sigma_omega

    def PoderParadorBetheModificado(self, Z, A, rho, E_kev, t):
        """Calcula o poder parador de Bethe modificado (com correção de J para baixas energias)

        conforme a Equação (4.103) do texto de referência.
        Retorna o valor em eV/Angstrom.
        o parametro t precisa ser um dicionario no formato:
        t_dict = {6: 0.77, 13: 0.815, 14: 0.822, 29: 0.83, 28: 0.83, 47: 0.852, 79: 0.851}
        """
        # Potencial médio de ionização (J) em eV (Adicionado o operador de multiplicação)
        if Z >= 13:
            J = 9.76 * Z + 58.5 / (Z**0.19)
        else:
            J = 11.5 * Z  # Aproximação para Z baixo

        # Parâmetro de ajuste 't' baseado na tabela da dissertação para alguns elementos

        #t_dict = {6: 0.77, 13: 0.815, 14: 0.822, 29: 0.83, 28: 0.83, 47: 0.852, 79: 0.851}
        t_dict = t
        t = t_dict.get(int(Z), 0.82)

        # Equação de Bethe modificada (Corrigido os operadores de multiplicação implícitos)
        termo_base = (785 * (rho * Z)) / (A * E_kev)
        argumento_log = (1.166 * (E_kev + t * J)) / J

        if argumento_log <= 0:
            return 0.0

        dE_dx = -termo_base * math.log(argumento_log)
        return dE_dx


class EDPsSolver():
    def __init__(self):
        self.derivadas = DerivadasSolver()


    def MetodoDiferencasFinitasLinear(self, p, q, r, a, b, alfa, beta, N):
        """Resolve um Problema de Valor de Contorno linear de 2ª ordem usando

        o Método das Diferenças Finitas.
        y'' + p(x)y' + q(x)y = r(x)
        com y(a) = alfa e y(b) = beta.

        :param p, q, r: Funções de x (podem ser lambdas)
        :param a, b: Limites do intervalo
        :param alfa, beta: Valores de contorno nas extremidades
        :param N: Número de pontos internos da malha
        :return: Tupla com os vetores x (malha) e y (solução)
        """
        h = (b - a) / (N + 1)
        x = np.linspace(a, b, N + 2)

        # Inicializa a matriz A (N x N) e o vetor b (N x 1)
        A = np.zeros((N, N))
        B = np.zeros(N)

        for i in range(N):
            xi = x[i + 1]  # Ponto interno atual

            # Coeficientes da discretização central
            # (1 - h/2 * p_i) * y_{i-1} - (2 + h^2 * q_i) * y_i + (1 + h/2 * p_i) * y_{i+1} = h^2 * r_i
            lower = 1.0 - (h / 2.0) * p(xi)
            diag = -2.0 + (h ** 2) * q(xi)
            upper = 1.0 + (h / 2.0) * p(xi)
            rhs = (h ** 2) * r(xi)

            # Preenchendo a matriz tridiagonal
            if i > 0:
                A[i, i - 1] = lower
            else:
                # O termo y_{0} passa para o lado direito multiplicando 'lower'
                rhs -= lower * alfa

            A[i, i] = diag

            if i < N - 1:
                A[i, i + 1] = upper
            else:
                # O termo y_{N+1} passa para o lado direito multiplicando 'upper'
                rhs -= upper * beta

            B[i] = rhs

        # Resolve o sistema linear A * y_interno = B
        y_interno = np.linalg.solve(A, B)

        # Monta a solução completa incluindo as condições de contorno nas pontas
        y_completo = np.zeros(N + 2)
        y_completo[0] = alfa
        y_completo[1:-1] = y_interno
        y_completo[-1] = beta

        return x, y_completo


    def MetodoDiferencasFinitasNaoLinear(self, f, fy, fyp, a, b, alfa, beta, N, tol=1e-6, max_iter=50):
        """Resolve um PVC não linear y'' = f(x, y, y') via Diferenças Finitas +

        Newton-Raphson.

        :param f: Função f(x, y, yp) que define y''
        :param fy: Derivada parcial df/dy(x, y, yp)
        :param fyp: Derivada parcial df/dyp(x, y, yp)
        :param a, b: Limites do intervalo
        :param alfa, beta: Condições de contorno y(a)=alfa, y(b)=beta
        :param N: Número de pontos internos da malha
        """
        h = (b - a) / (N + 1)
        x = np.linspace(a, b, N + 2)

        # Estimativa inicial linear simples entre (a, alfa) e (b, beta)
        y = alfa + (x - a) * (beta - alfa) / (b - a)

        for it in range(max_iter):
            F = np.zeros(N)
            J = np.zeros((N, N))

            for i in range(N):
                idx = i + 1  # Índice no vetor global de x e y
                xi = x[idx]
                yi = y[idx]

                # Aproximações de y' e y'' por diferenças centradas
                yp_i = (y[idx + 1] - y[idx - 1]) / (2 * h)
                ypp_i = (y[idx + 1] - 2 * yi + y[idx - 1]) / (h**2)

                # Vetor residual F_i = y'' - f(x, y, y')
                F[i] = ypp_i - f(xi, yi, yp_i)

                # Elementos da Matriz Jacobiana
                df_dy = fy(xi, yi, yp_i)
                df_dyp = fyp(xi, yi, yp_i)

                # Derivadas parciais em relação a y_{i-1}, y_i, y_{i+1}
                lower = 1.0 / (h**2) + df_dyp / (2 * h)
                diag = -2.0 / (h**2) - df_dy
                upper = 1.0 / (h**2) - df_dyp / (2 * h)

                if i > 0:
                    J[i, i - 1] = lower
                J[i, i] = diag
                if i < N - 1:
                    J[i, i + 1] = upper

            # Resolver J * delta_y = -F
            delta_y = np.linalg.solve(J, -F)
            y[1:-1] += delta_y

            # Critério de parada (norma da correção)
            if np.linalg.norm(delta_y, ord=np.inf) < tol:
                break

        return x, y

    def ResolverEDPDiscretizada(
        self,
        residuos_equacao,
        campo_inicial,
        condicoes_contorno=(),
        solucionador=None,
        escala_equacao=1.0,
        escala_contorno=1.0,
        **opcoes_solver,
    ):
        """Resolve EDPs genericas apos discretizacao fornecida pelo usuario.

        residuos_equacao(campo) retorna os residuos discretizados da EDP;
        cada condicao_contorno(campo) retorna residuos de contorno, interface,
        observacao ou outra restricao. `campo` conserva o shape de campo_inicial.
        Assim o usuario escolhe a malha, operadores e tipo de contorno, podendo
        reutilizar DerivadasSolver, FuncoesSolver ou outros operadores Omega.

        `solucionador` pode implementar ResolverSistemaNaoLinear(residuo, chute,
        **opcoes_solver); por padrao usa FuncoesSolverNaoLinear. A solucao e
        numerica/local: nao garante unicidade nem existencia e depende da
        discretizacao, do condicionamento, do chute e da consistencia das escalas.
        """
        if not callable(residuos_equacao):
            raise TypeError("residuos_equacao deve ser uma funcao do campo discretizado.")
        campo = np.asarray(campo_inicial, dtype=float)
        if campo.size == 0 or not np.all(np.isfinite(campo)):
            raise ValueError("campo_inicial deve conter valores finitos e nao vazios.")
        forma = campo.shape
        if callable(condicoes_contorno):
            condicoes = [condicoes_contorno]
        else:
            try:
                condicoes = list(condicoes_contorno)
            except TypeError:
                raise TypeError("condicoes_contorno deve ser uma funcao ou sequencia de funcoes.") from None
        if not all(callable(condicao) for condicao in condicoes):
            raise TypeError("Cada condicao de contorno deve ser uma funcao do campo.")

        def escala_positiva(valor, nome):
            numero = float(valor)
            if not math.isfinite(numero) or numero <= 0.0:
                raise ValueError(f"{nome} deve ser positiva e finita.")
            return numero

        escala_eq = escala_positiva(escala_equacao, "escala_equacao")
        escala_bc = escala_positiva(escala_contorno, "escala_contorno")

        def avaliar_grupo(funcao, estado, nome):
            valores = np.asarray(funcao(estado.copy()), dtype=float).reshape(-1)
            if valores.size == 0 or not np.all(np.isfinite(valores)):
                raise ValueError(f"{nome} deve retornar residuos finitos e nao vazios.")
            return valores

        def residuos_totais(incognitas):
            estado = np.asarray(incognitas, dtype=float).reshape(forma)
            partes = [escala_eq * avaliar_grupo(residuos_equacao, estado, "residuos_equacao")]
            partes.extend(
                escala_bc * avaliar_grupo(condicao, estado, "residuos_contorno")
                for condicao in condicoes
            )
            return np.concatenate(partes)

        if solucionador is None:
            from funcoes.funcoes import FuncoesSolverNaoLinear

            solucionador = FuncoesSolverNaoLinear()
        metodo = getattr(solucionador, "ResolverSistemaNaoLinear", None)
        if not callable(metodo):
            raise TypeError("solucionador deve implementar ResolverSistemaNaoLinear.")

        resultado = metodo(residuos_totais, campo.reshape(-1), **opcoes_solver)
        campo_final = np.asarray(resultado["solucao"], dtype=float).reshape(forma)
        resultado["campo"] = campo_final
        resultado["residuos_equacao"] = avaliar_grupo(
            residuos_equacao, campo_final, "residuos_equacao"
        )
        resultado["residuos_contorno"] = [
            avaliar_grupo(condicao, campo_final, "residuos_contorno")
            for condicao in condicoes
        ]
        resultado["escalas"] = {
            "equacao": escala_eq,
            "contorno": escala_bc,
        }
        return resultado

    @staticmethod
    def _coordenadas_finitas(coordenadas):
        bruto = np.asarray(coordenadas)
        if np.iscomplexobj(bruto):
            raise ValueError("coordenadas deve conter valores reais.")
        ponto = np.asarray(bruto, dtype=float)
        if ponto.ndim != 1 or ponto.size == 0 or not np.all(np.isfinite(ponto)):
            raise ValueError("coordenadas deve ser um vetor nao vazio de valores finitos.")
        return ponto

    @staticmethod
    def _avaliar_metrica(metrica, coordenadas, cache=None):
        chave = tuple(float(valor) for valor in coordenadas)
        if cache is not None and chave in cache:
            return cache[chave]
        try:
            bruto = metrica(*chave)
            if np.iscomplexobj(bruto):
                raise ValueError
            g = np.asarray(bruto, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("metrica deve retornar uma matriz numerica.") from None
        n = len(chave)
        if g.shape != (n, n) or not np.all(np.isfinite(g)):
            raise ValueError(f"metrica deve retornar uma matriz finita de shape ({n}, {n}).")
        tolerancia_simetria = (
            100.0 * np.finfo(float).eps
            * max(float(np.linalg.norm(g, ord=np.inf)), np.finfo(float).tiny)
        )
        if not np.allclose(g, g.T, rtol=0.0, atol=tolerancia_simetria):
            raise ValueError("A metrica deve ser simetrica.")
        try:
            inversa = np.linalg.solve(g, np.eye(n))
        except np.linalg.LinAlgError as erro:
            raise ValueError("A metrica e singular no ponto avaliado.") from erro
        if not np.all(np.isfinite(inversa)):
            raise ValueError("A inversa numerica da metrica nao e finita.")
        if cache is not None:
            cache[chave] = g
        return g

    def SimbolosChristoffel(self, metrica, coordenadas, derivadas=None, _cache=None):
        """Calcula numericamente os simbolos de Christoffel da metrica.

        ``metrica(*coords)`` deve devolver uma matriz simetrica, finita e
        nao singular de shape ``(n, n)``, onde ``n=len(coordenadas)``. A
        assinatura historica em quatro coordenadas continua valida.

        Retorna ``Gamma[rho, mu, nu]`` para a conexao de Levi-Civita:
        1/2 g^(rho, lam) (d_mu g_(lam,nu) + d_nu g_(lam,mu)
                            - d_lam g_(mu,nu)).
        As derivadas sao aproximacoes numericas de cinco pontos; ajuste o
        passo de ``DerivadasSolver`` a escala e suavidade da metrica. A metrica
        deve ser ao menos C1 no stencil numerico e nao degenerada em toda a
        vizinhanca amostrada.
        """
        if not callable(metrica):
            raise TypeError("metrica deve ser uma funcao das coordenadas.")
        ponto = self._coordenadas_finitas(coordenadas)
        calculador_derivadas = self.derivadas if derivadas is None else derivadas
        if not isinstance(calculador_derivadas, DerivadasSolver):
            raise TypeError("derivadas deve ser uma instancia de DerivadasSolver.")

        cache = {} if _cache is None else _cache
        n = ponto.size
        g = self._avaliar_metrica(metrica, ponto, cache)
        inversa = np.linalg.solve(g, np.eye(n))
        derivadas_metrica = np.empty((n, n, n), dtype=float)
        for eixo in range(n):
            for i in range(n):
                for j in range(n):
                    def componente(valor, eixo=eixo, i=i, j=j):
                        argumentos = ponto.copy()
                        argumentos[eixo] = valor
                        return self._avaliar_metrica(metrica, argumentos, cache)[i, j]

                    derivadas_metrica[eixo, i, j] = calculador_derivadas.Derivada(
                        componente, ponto[eixo]
                    )

        gamma = 0.5 * (
            np.einsum("rl,mln->rmn", inversa, derivadas_metrica)
            + np.einsum("rl,nlm->rmn", inversa, derivadas_metrica)
            - np.einsum("rl,lmn->rmn", inversa, derivadas_metrica)
        )
        return 0.5 * (gamma + np.swapaxes(gamma, 1, 2))

    @staticmethod
    def _avaliar_christoffel(simbolos_christoffel, coordenadas, cache):
        chave = tuple(float(valor) for valor in coordenadas)
        if chave in cache:
            return cache[chave]
        n = len(chave)
        try:
            bruto = simbolos_christoffel(*chave)
            if np.iscomplexobj(bruto):
                raise ValueError
            gamma = np.asarray(bruto, dtype=float)
        except (TypeError, ValueError):
            gamma = None
        if gamma is None or gamma.shape != (n, n, n):
            try:
                gamma = np.empty((n, n, n), dtype=float)
                for rho in range(n):
                    for mu in range(n):
                        for nu in range(n):
                            gamma[rho, mu, nu] = simbolos_christoffel(
                                rho, mu, nu, *chave
                            )
            except (TypeError, ValueError, IndexError):
                raise ValueError(
                    "Christoffel deve aceitar *coords e retornar shape (n, n, n), "
                    "ou aceitar (rho, mu, nu, *coords)."
                ) from None
        if not np.all(np.isfinite(gamma)):
            raise ValueError("Os simbolos de Christoffel devem ser finitos no stencil.")
        cache[chave] = gamma
        return gamma

    def TensorRiemann(self, simbolos_christoffel, coordenadas, derivadas=None):
        """Calcula R^rho_(sigma mu nu) a partir de uma conexao Gamma.

        Aceita uma funcao ``Gamma(*coords)`` que retorna um array ``(n,n,n)``
        ou uma funcao componente ``Gamma(rho, mu, nu, *coords)``. A conexao
        deve ser diferenciavel no stencil; para uma conexao de Levi-Civita,
        isso requer uma metrica pelo menos C2. Usa a
        convencao R^rho_(sigma mu nu) = d_mu Gamma^rho_(nu sigma)
        - d_nu Gamma^rho_(mu sigma) + Gamma^rho_(mu lam)Gamma^lam_(nu sigma)
        - Gamma^rho_(nu lam)Gamma^lam_(mu sigma). As derivadas sao numericas.
        """
        if not callable(simbolos_christoffel):
            raise TypeError("simbolos_christoffel deve ser uma funcao.")
        ponto = self._coordenadas_finitas(coordenadas)
        calculador_derivadas = self.derivadas if derivadas is None else derivadas
        if not isinstance(calculador_derivadas, DerivadasSolver):
            raise TypeError("derivadas deve ser uma instancia de DerivadasSolver.")
        n = ponto.size
        cache = {}
        gamma = self._avaliar_christoffel(simbolos_christoffel, ponto, cache)
        derivada_gamma = np.empty((n, n, n, n), dtype=float)
        for eixo in range(n):
            for rho in range(n):
                for mu in range(n):
                    for nu in range(n):
                        def componente(valor, eixo=eixo, rho=rho, mu=mu, nu=nu):
                            argumentos = ponto.copy()
                            argumentos[eixo] = valor
                            return self._avaliar_christoffel(
                                simbolos_christoffel, argumentos, cache
                            )[rho, mu, nu]

                        derivada_gamma[eixo, rho, mu, nu] = (
                            calculador_derivadas.Derivada(componente, ponto[eixo])
                        )

        tensor = (
            derivada_gamma.transpose(1, 3, 0, 2)
            - derivada_gamma.transpose(1, 3, 2, 0)
            + np.einsum("rml,lns->rsmn", gamma, gamma)
            - np.einsum("rnl,lms->rsmn", gamma, gamma)
        )
        if not np.all(np.isfinite(tensor)):
            raise FloatingPointError("O tensor de Riemann nao e finito.")
        return tensor

    def TensorRiemannDaMetrica(self, metrica, coordenadas, derivadas=None):
        """Calcula Riemann a partir de uma metrica arbitraria nao degenerada."""
        if not callable(metrica):
            raise TypeError("metrica deve ser uma funcao das coordenadas.")
        ponto = self._coordenadas_finitas(coordenadas)
        calculador_derivadas = self.derivadas if derivadas is None else derivadas
        if not isinstance(calculador_derivadas, DerivadasSolver):
            raise TypeError("derivadas deve ser uma instancia de DerivadasSolver.")
        cache_metrica = {}
        cache_gamma = {}

        def gamma(*argumentos):
            chave = tuple(float(valor) for valor in argumentos)
            if chave not in cache_gamma:
                cache_gamma[chave] = self.SimbolosChristoffel(
                    metrica, chave, calculador_derivadas, _cache=cache_metrica
                )
            return cache_gamma[chave]

        return self.TensorRiemann(gamma, ponto, calculador_derivadas)

    def EquacoesRicciVacioDaMetrica(self, metrica, coordenadas, derivadas=None):
        """Calcula as componentes independentes de Ricci=0 diretamente da metrica.

        As derivadas de Christoffel exigem derivadas numericas de segunda ordem
        da metrica. Para uma metrica em n dimensoes, retorna n(n+1)/2 residuos;
        em quatro dimensoes sao dez. Ajuste DerivadasSolver(h=...) a escala e
        suavidade da metrica.
        """
        calculador_derivadas = self.derivadas if derivadas is None else derivadas
        ponto = np.asarray(coordenadas, dtype=float)
        cache_metrica = {}
        cache_gamma = {}

        def gamma(*argumentos):
            chave = tuple(float(valor) for valor in argumentos)
            if chave not in cache_gamma:
                cache_gamma[chave] = self.SimbolosChristoffel(
                    metrica, argumentos, calculador_derivadas, _cache=cache_metrica
                )
            return cache_gamma[chave]

        return self.EquacoesRicciVacio(gamma, ponto, calculador_derivadas)

    def ResolverMetricaVacio(
        self,
        modelo_metrica,
        parametros_iniciais,
        pontos_colocacao,
        condicoes_dados,
        derivadas=None,
        tolerancia=1e-8,
        max_iter=50,
        amortecimento=1e-3,
        passo_jacobiano=1e-5,
        escala_ricci=1.0,
        escala_dados=1.0,
    ):
        """Ajusta uma metrica parametrizada aos residuos R_mn=0 e aos dados.

        modelo_metrica(parametros, x0, x1, x2, x3) deve retornar g_mn (4x4).
        pontos_colocacao e uma sequencia de pontos de coordenadas (cada um com
        quatro valores); em cada ponto sao impostas as dez equacoes Ricci=0.
        condicoes_dados e callables condicao(parametros) que retornam residuos
        de contorno, observacoes ou restricoes escolhidos pelo usuario. E preciso
        pelo menos uma condicao para reduzir a liberdade de gauge e fixar dados.

        Resolve em parametros finitos por Gauss-Newton amortecido (Levenberg-
        Marquardt), com Jacobiano por diferencas centrais. Retorna o melhor
        conjunto encontrado, mesmo se convergiu=False; isso nao garante uma
        solucao unica, global, Lorentziana ou independente da coordenada.
        escala_ricci e escala_dados normalizam os dois grupos de residuos para
        que suas unidades e magnitudes nao desequilibrem o ajuste.
        """
        if not callable(modelo_metrica):
            raise TypeError("modelo_metrica deve ser uma funcao parametrizada.")
        parametros = np.asarray(parametros_iniciais, dtype=float)
        pontos = np.asarray(pontos_colocacao, dtype=float)
        if parametros.ndim != 1 or parametros.size == 0 or not np.all(np.isfinite(parametros)):
            raise ValueError("parametros_iniciais deve ser um vetor nao vazio e finito.")
        if pontos.ndim == 1:
            pontos = pontos.reshape(1, -1)
        if pontos.ndim != 2 or pontos.shape[1] != 4 or pontos.shape[0] == 0:
            raise ValueError("pontos_colocacao deve ter shape (n, 4), com n >= 1.")
        if not np.all(np.isfinite(pontos)):
            raise ValueError("pontos_colocacao deve conter coordenadas finitas.")
        if callable(condicoes_dados):
            condicoes = [condicoes_dados]
        else:
            try:
                condicoes = list(condicoes_dados)
            except TypeError:
                raise ValueError("condicoes_dados deve ser callable ou sequencia de callables.") from None
        if not condicoes or not all(callable(condicao) for condicao in condicoes):
            raise ValueError("Forneca ao menos uma funcao de condicao/observacao.")

        calculador_derivadas = self.derivadas if derivadas is None else derivadas
        if not isinstance(calculador_derivadas, DerivadasSolver):
            raise TypeError("derivadas deve ser uma instancia de DerivadasSolver.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter deve ser um inteiro positivo.")

        def validar_positivo(valor, nome):
            numero = float(valor)
            if not math.isfinite(numero) or numero <= 0.0:
                raise ValueError(f"{nome} deve ser positivo e finito.")
            return numero

        tolerancia = validar_positivo(tolerancia, "tolerancia")
        amortecimento = validar_positivo(amortecimento, "amortecimento")
        passo_jacobiano = validar_positivo(passo_jacobiano, "passo_jacobiano")
        escala_ricci = validar_positivo(escala_ricci, "escala_ricci")
        escala_dados = validar_positivo(escala_dados, "escala_dados")

        def avaliar_residuos(valores):
            residuos_ricci = []
            for ponto in pontos:
                funcao_metrica = lambda *coords, valores=valores: modelo_metrica(
                    valores.copy(), *coords
                )
                componentes = self.EquacoesRicciVacioDaMetrica(
                    funcao_metrica, ponto, calculador_derivadas
                )
                residuos_ricci.extend(componentes.values())
            residuos_dados = []
            for condicao in condicoes:
                bruto = np.asarray(condicao(valores.copy()), dtype=float).reshape(-1)
                if bruto.size == 0 or not np.all(np.isfinite(bruto)):
                    raise ValueError("Cada condicao deve retornar residuos numericos finitos.")
                residuos_dados.extend(bruto.tolist())
            ricci = np.asarray(residuos_ricci, dtype=float)
            dados = np.asarray(residuos_dados, dtype=float)
            composto = np.concatenate((escala_ricci * ricci, escala_dados * dados))
            return composto, ricci, dados

        residuo, residuo_ricci, residuo_dados = avaliar_residuos(parametros)
        tamanho_residuo = residuo.size
        norma = float(np.linalg.norm(residuo))
        convergiu = norma <= tolerancia
        iteracoes = 0
        mensagem = "tolerancia de residuos atingida" if convergiu else "maximo de iteracoes nao atingido"

        while not convergiu and iteracoes < max_iter:
            jacobiano = np.empty((tamanho_residuo, parametros.size), dtype=float)
            for indice in range(parametros.size):
                passo = passo_jacobiano * max(1.0, abs(parametros[indice]))
                mais = parametros.copy()
                menos = parametros.copy()
                mais[indice] += passo
                menos[indice] -= passo
                try:
                    residuo_mais = avaliar_residuos(mais)[0]
                except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                    residuo_mais = None
                try:
                    residuo_menos = avaliar_residuos(menos)[0]
                except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                    residuo_menos = None
                if residuo_mais is not None and residuo_menos is not None:
                    jacobiano[:, indice] = (residuo_mais - residuo_menos) / (2.0 * passo)
                elif residuo_mais is not None:
                    jacobiano[:, indice] = (residuo_mais - residuo) / passo
                elif residuo_menos is not None:
                    jacobiano[:, indice] = (residuo - residuo_menos) / passo
                else:
                    raise ValueError(
                        f"Nao foi possivel perturbar o parametro {indice} sem invalidar a metrica."
                    )

            diagonal = np.maximum(np.sum(jacobiano**2, axis=0), 1.0)
            regularizador = np.diag(np.sqrt(amortecimento * diagonal))
            sistema_aumentado = np.vstack((jacobiano, regularizador))
            residuo_aumentado = np.concatenate(
                (-residuo, np.zeros(parametros.size, dtype=float))
            )
            try:
                incremento = np.linalg.lstsq(
                    sistema_aumentado, residuo_aumentado, rcond=None
                )[0]
            except np.linalg.LinAlgError:
                amortecimento *= 10.0
                iteracoes += 1
                continue

            candidato = parametros + incremento
            try:
                residuo_candidato, ricci_candidato, dados_candidato = avaliar_residuos(candidato)
                norma_candidata = float(np.linalg.norm(residuo_candidato))
            except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                norma_candidata = math.inf

            iteracoes += 1
            if norma_candidata < norma:
                parametros = candidato
                residuo = residuo_candidato
                residuo_ricci = ricci_candidato
                residuo_dados = dados_candidato
                norma = norma_candidata
                amortecimento = max(amortecimento / 3.0, 1e-12)
                convergiu = norma <= tolerancia
                if convergiu:
                    mensagem = "tolerancia de residuos atingida"
                elif np.linalg.norm(incremento) <= tolerancia * (1.0 + np.linalg.norm(parametros)):
                    mensagem = "incremento pequeno antes de atingir a tolerancia"
                    break
            else:
                amortecimento *= 10.0
                if amortecimento > 1e12:
                    mensagem = "amortecimento maximo atingido; o ajuste estagnou"
                    break

        return {
            "parametros": parametros,
            "convergiu": convergiu,
            "iteracoes": iteracoes,
            "norma_residuo": norma,
            "max_residuo_ricci": float(np.max(np.abs(residuo_ricci))),
            "max_residuo_dados": float(np.max(np.abs(residuo_dados))),
            "amortecimento_final": amortecimento,
            "mensagem": mensagem,
            "limitacoes": [
                "A solucao fica restrita ao modelo parametrico fornecido pelo usuario.",
                "Ajuste local nao linear: resultado depende do chute inicial e pode nao ser unico.",
                "Condicoes fornecidas devem fixar dados/gauge; verifique assinatura e nao degenerescencia.",
                "As derivadas espaciais sao numericas e o ajuste nao prova existencia ou unicidade global.",
            ],
        }

    def TensorRicci(self, simbolos_christoffel, coordenadas, derivadas=None):
        """Calcula R_mn=R^rho_(m rho n) para uma conexao em n dimensoes."""
        riemann = self.TensorRiemann(
            simbolos_christoffel, coordenadas, derivadas
        )
        return np.einsum("rmrn->mn", riemann)

    def TensorRicciDaMetrica(self, metrica, coordenadas, derivadas=None):
        """Calcula o tensor de Ricci covariante diretamente da metrica."""
        riemann = self.TensorRiemannDaMetrica(metrica, coordenadas, derivadas)
        return np.einsum("rmrn->mn", riemann)

    def EscalarRicci(self, metrica, coordenadas, derivadas=None):
        """Calcula R=g^mn R_mn para uma metrica nao degenerada."""
        ponto = self._coordenadas_finitas(coordenadas)
        g = self._avaliar_metrica(metrica, ponto)
        ricci = self.TensorRicciDaMetrica(metrica, ponto, derivadas)
        return float(np.einsum("mn,mn->", np.linalg.solve(g, np.eye(len(ponto))), ricci))

    def TensorEinstein(self, metrica, coordenadas, derivadas=None):
        """Calcula G_mn=R_mn-(1/2)g_mn R, sem constante cosmologica."""
        ponto = self._coordenadas_finitas(coordenadas)
        g = self._avaliar_metrica(metrica, ponto)
        ricci = self.TensorRicciDaMetrica(metrica, ponto, derivadas)
        escalar = float(np.einsum("mn,mn->", np.linalg.solve(g, np.eye(len(ponto))), ricci))
        return ricci - 0.5 * g * escalar

    def DerivadaCovariante(
        self, metrica, campo_tensor, coordenadas, variancia_indices, derivadas=None
    ):
        """Calcula nabla_lambda T para qualquer tensor em coordenadas.

        ``campo_tensor(*coords)`` retorna um array cuja forma e ``(n,)*rank``.
        ``variancia_indices`` informa a ordem dos indices como ``'contravariante'``
        ou ``'covariante'`` (tambem aceita ``'up'``/``'down'``). O indice da
        derivada e acrescentado como ultimo eixo do resultado. Para um escalar,
        use uma sequencia vazia. Cada derivada parcial e numerica; a conexao e a
        de Levi-Civita derivada da metrica, que deve ser nao degenerada e
        suficientemente suave no stencil (campo C1 e metrica C1).
        """
        if not callable(metrica) or not callable(campo_tensor):
            raise TypeError("metrica e campo_tensor devem ser funcoes das coordenadas.")
        ponto = self._coordenadas_finitas(coordenadas)
        if isinstance(variancia_indices, str):
            raise TypeError("variancia_indices deve ser uma sequencia, inclusive vazia para escalar.")
        try:
            variancias = tuple(variancia_indices)
        except TypeError:
            raise TypeError("variancia_indices deve ser uma sequencia de tipos de indice.") from None
        aliases = {
            "contravariante": "up",
            "covariante": "down",
            "up": "up",
            "down": "down",
        }
        try:
            tipos = tuple(aliases[tipo] for tipo in variancias)
        except (KeyError, TypeError):
            raise ValueError(
                "Cada indice deve ser 'contravariante'/'covariante' ou 'up'/'down'."
            ) from None
        calculador_derivadas = self.derivadas if derivadas is None else derivadas
        if not isinstance(calculador_derivadas, DerivadasSolver):
            raise TypeError("derivadas deve ser uma instancia de DerivadasSolver.")
        n = ponto.size
        cache_tensor = {}

        def avaliar_tensor(argumentos):
            chave = tuple(float(valor) for valor in argumentos)
            if chave not in cache_tensor:
                try:
                    bruto = campo_tensor(*chave)
                    if np.iscomplexobj(bruto):
                        raise ValueError
                    valor = np.asarray(bruto, dtype=float)
                except (TypeError, ValueError):
                    raise ValueError("campo_tensor deve retornar um tensor numerico.") from None
                if valor.shape != (n,) * len(tipos) or not np.all(np.isfinite(valor)):
                    raise ValueError(
                        f"campo_tensor deve retornar valores finitos de shape {(n,) * len(tipos)}."
                    )
                cache_tensor[chave] = valor
            return cache_tensor[chave]

        tensor = avaliar_tensor(ponto)
        gamma = self.SimbolosChristoffel(metrica, ponto, calculador_derivadas)
        parcial = np.empty(tensor.shape + (n,), dtype=float)
        for eixo in range(n):
            for indices in np.ndindex(tensor.shape):
                def componente(valor, eixo=eixo, indices=indices):
                    argumentos = ponto.copy()
                    argumentos[eixo] = valor
                    return avaliar_tensor(argumentos)[indices]

                parcial[indices + (eixo,)] = calculador_derivadas.Derivada(
                    componente, ponto[eixo]
                )

        resultado = parcial.copy()
        for indices in np.ndindex(tensor.shape):
            for eixo in range(n):
                valor = parcial[indices + (eixo,)]
                for posicao, tipo in enumerate(tipos):
                    indice = indices[posicao]
                    for somado in range(n):
                        indices_troca = (
                            indices[:posicao] + (somado,) + indices[posicao + 1 :]
                        )
                        if tipo == "up":
                            valor += gamma[indice, eixo, somado] * tensor[indices_troca]
                        else:
                            valor -= gamma[somado, eixo, indice] * tensor[indices_troca]
                resultado[indices + (eixo,)] = valor
        if not np.all(np.isfinite(resultado)):
            raise FloatingPointError("A derivada covariante nao e finita.")
        return resultado

    def Geodesica(
        self,
        metrica,
        coordenadas_iniciais,
        velocidades_iniciais,
        parametro_inicial,
        parametro_final,
        derivadas=None,
        passo_inicial=None,
        tolerancia_absoluta=1e-9,
        tolerancia_relativa=1e-7,
        max_passos=100000,
    ):
        """Integra uma geodesica afim da metrica usando o RK45 do projeto.

        Integra ``x'^mu=v^mu`` e ``v'^mu=-Gamma^mu_(alpha beta)v^alpha v^beta``
        para parametro afim e metrica suficientemente suave e nao degenerada.
        Retorna parametro, coordenadas, velocidades, estados completos,
        diagnosticos do RK45 e a norma ``g_mn v^m v^n`` ao longo da solucao.
        Essa norma e uma verificacao de consistencia, nao uma correcao imposta.
        A curvatura usa derivadas numericas de cinco pontos e RK45 controla erro
        local de integracao, nao fornece uma garantia de erro global.
        """
        if not callable(metrica):
            raise TypeError("metrica deve ser uma funcao das coordenadas.")
        posicao = self._coordenadas_finitas(coordenadas_iniciais)
        velocidade = self._coordenadas_finitas(velocidades_iniciais)
        if velocidade.shape != posicao.shape:
            raise ValueError("coordenadas_iniciais e velocidades_iniciais devem ter shape igual.")
        inicio = float(parametro_inicial)
        fim = float(parametro_final)
        if not math.isfinite(inicio) or not math.isfinite(fim):
            raise ValueError("Os parametros afins devem ser finitos.")
        calculador_derivadas = self.derivadas if derivadas is None else derivadas
        if not isinstance(calculador_derivadas, DerivadasSolver):
            raise TypeError("derivadas deve ser uma instancia de DerivadasSolver.")

        n = posicao.size

        def equacao_geodesica(_, estado):
            x = estado[:n]
            v = estado[n:]
            gamma = self.SimbolosChristoffel(metrica, x, calculador_derivadas)
            aceleracao = -np.einsum("mab,a,b->m", gamma, v, v)
            return np.concatenate((v, aceleracao))

        solver_edo = EDOsSolver()
        parametros, estados_brutos, diagnosticos = solver_edo.Rk45(
            equacao_geodesica,
            np.concatenate((posicao, velocidade)),
            inicio,
            fim,
            passo_inicial=passo_inicial,
            tolerancia_absoluta=tolerancia_absoluta,
            tolerancia_relativa=tolerancia_relativa,
            max_passos=max_passos,
        )
        estados = np.asarray(estados_brutos, dtype=float).reshape((-1, 2 * n))
        coordenadas = estados[:, :n]
        velocidades = estados[:, n:]
        normas = np.empty(len(estados), dtype=float)
        for indice, (x, v) in enumerate(zip(coordenadas, velocidades)):
            g = self._avaliar_metrica(metrica, x)
            normas[indice] = float(v @ g @ v)
        if not np.all(np.isfinite(normas)):
            raise FloatingPointError("A norma da velocidade geodesica nao e finita.")
        return {
            "parametros": np.asarray(parametros, dtype=float),
            "coordenadas": coordenadas,
            "velocidades": velocidades,
            "estados": estados,
            "norma_velocidade_inicial": float(normas[0]),
            "norma_velocidade": normas,
            "diagnosticos": diagnosticos,
        }

    def EquacoesRicciVacio(self, simbolos_christoffel, coordenadas, derivadas=None):
        """Retorna os residuos independentes R_mn=0 para Ricci simetrico.

        O resultado e um dicionario com chaves (mu, nu), para mu <= nu, e
        valores R_mn avaliados no ponto; ha n(n+1)/2 componentes em n dimensoes
        (dez em quatro dimensoes). A simetria e propria da conexao de Levi-Civita.
        Residuos proximos de zero indicam satisfacao local das equacoes na
        precisao das diferencas finitas utilizadas.
        """
        ricci = self.TensorRicci(simbolos_christoffel, coordenadas, derivadas)
        return {
            (mu, nu): float(ricci[mu, nu])
            for mu in range(4)
            for nu in range(mu, 4)
        }

    def EquacaoDaOnda1D(self, v, function, x, y):
        """
        Equação de Onda em 1D utilizando Derivadas parciais de segunda ordem\n
        v: é a velocidade\n
        function: é a função de onda. Uma função que pode ser definida pelo usuario.\n
        x e y: São as coordenadas pontuais unidimencionais
        """
        d2x = d.DerivadaSegundaX(function, x, y)
        d2y = d.DerivadaSegundaY(function, x, y)
        return d2y - (v**2) * d2x

    def EquacaoDaOnda3D(self, v, function, x, y, z, t):
        """
        Equação de Onda em 3D utilizando Derivadas parciais de segunda ordem\n
        v: é a velocidade\n
        function: é a função de onda com 4 parâmetros f(x, y, z, t).\n
        x, y, z: Coordenadas espaciais\n
        t: Coordenada temporal
        """
        from derivadas.derivadas import DerivadasSolver
        d = DerivadasSolver()

        # 1. Laplaciano espacial: varia x, y, z mantendo o tempo 't' fixo
        Laplaciano = d.Laplaciano(
            lambda x_val, y_val, z_val: function(x_val, y_val, z_val, t), 
            x, y, z
        )
        
        # 2. Derivada segunda temporal: varia apenas o tempo 't' mantendo x, y, z fixos
        d2t = d.SegundaDerivadaCentralizada(
            lambda t_val: function(x, y, z, t_val), 
            t
        )

        return d2t - (v**2) * Laplaciano

    def Arrasto3D(self, func_u, func_v, func_w, func_p, x, y, z, rho, nu, fx=0.0, fy=0.0, fz=0.0):
        """
        Calcula o arrasto aerodinâmico em três dimensões (3D)
        utilizando a equação de Navier-Stokes Convectiva em 3D.
        
        Parâmetros:
            - func_u, func_v, func_w, func_p: Funções de velocidade e pressão.
            - x, y, z: Variáveis espaciais (tf.Variable).
            - rho, nu: Densidade e viscosidade do fluido.
            - fx, fy, fz: Forças externas (recomenda-se passar o módulo de arrasto aerodinâmico).
        """
        Navie = EDOsSolver()
        # Aqui estava o erro: passava 0.0 fixo. Agora repassa os argumentos corretos: fx, fy, fz
        return Navie.NavierStokesConvectiva3D(func_u, func_v, func_w, func_p, x, y, z, rho, nu, fx=fx, fy=fy, fz=fz)

class TeoremasSolver():
    """
    Solver numérico para os grandes teoremas do cálculo vetorial.
    Calcula independentemente ambos os lados de cada teorema usando
    derivadas numéricas (5 pontos) e integração numérica (Simpson composto),
    sem recorrer a simplificações analíticas.
    """
    def __init__(self, n_passos=100):
        """
        Inicializa o solver.
        Parâmetros:
        - n_passos: número de subintervalos para a integração composta (deve ser par).
        """
        self.n_passos = n_passos if n_passos % 2 == 0 else n_passos + 1
        self.derivadas = DerivadasSolver()

    def _integrar_simpson_1d(self, f, a, b):
        """Regra de Simpson composta 1D."""
        n = self.n_passos
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([float(f(xi)) for xi in x])
        soma = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
        return (h / 3.0) * soma

    def _integrar_simpson_2d(self, f, x_limits, y_limits):
        """Regra de Simpson composta 2D (aninhada)."""
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        def integrando_x(x):
            return self._integrar_simpson_1d(lambda y: f(x, y), y_min, y_max)
        return self._integrar_simpson_1d(integrando_x, x_min, x_max)

    def _integrar_simpson_3d(self, f, x_limits, y_limits, z_limits):
        """Regra de Simpson composta 3D (aninhada)."""
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        z_min, z_max = z_limits
        def integrando_x(x):
            def integrando_y(y):
                return self._integrar_simpson_1d(lambda z: f(x, y, z), z_min, z_max)
            return self._integrar_simpson_1d(integrando_y, y_min, y_max)
        return self._integrar_simpson_1d(integrando_x, x_min, x_max)

    def CalcularTeoremaDeGreen(self, P, Q, curva, t_limits, regiao_limits):
        """
        Valida numericamente o Teorema de Green:
        ∮_C (P dx + Q dy) = ∬_D (∂Q/∂x - ∂P/∂y) dA

        Parâmetros:
        - P, Q: Funções escalares P(x, y) e Q(x, y).
        - curva: Função que recebe t e retorna [x(t), y(t)].
        - t_limits: Tupla (t_inicio, t_fim) para a curva fechada C.
        - regiao_limits: Tupla ((x_min, x_max), (y_min, y_max)) definindo a região retangular D.

        Retorna:
        - Dicionário com 'integral_linha', 'integral_dupla' e 'diferenca_absoluta'.
        """
        t_inicio, t_fim = t_limits
        x_limits, y_limits = regiao_limits

        # --- LADO ESQUERDO: Integral de Linha ---
        def integrando_linha(t):
            r = curva(t)
            x, y = r[0], r[1]
            # Derivadas numéricas da parametrização da curva
            dx_dt = self.derivadas.DerivadaCentralizada(lambda tau: curva(tau)[0], t)
            dy_dt = self.derivadas.DerivadaCentralizada(lambda tau: curva(tau)[1], t)
            return P(x, y) * dx_dt + Q(x, y) * dy_dt
        
        lado_esquerdo = self._integrar_simpson_1d(integrando_linha, t_inicio, t_fim)

        # --- LADO DIREITO: Integral Dupla do Rotacional 2D ---
        def integrando_dupla(x, y):
            # ∂Q/∂x
            dQ_dx = self.derivadas.DerivadaCentralizada(lambda x_val: Q(x_val, y), x)
            # ∂P/∂y
            dP_dy = self.derivadas.DerivadaCentralizada(lambda y_val: P(x, y_val), y)
            return dQ_dx - dP_dy
        
        lado_direito = self._integrar_simpson_2d(integrando_dupla, x_limits, y_limits)

        return {
            "teorema": "Green",
            "integral_linha": lado_esquerdo,
            "integral_dupla": lado_direito,
            "diferenca_absoluta": abs(lado_esquerdo - lado_direito)
        }

    def CalcularTeoremaDeStokes(self, F, superficie, u_limits, v_limits, curva_bordo, t_limits):
        """
        Valida numericamente o Teorema de Stokes:
        ∮_C F · dr = ∬_S (∇ × F) · dS

        Parâmetros:
        - F: Função que recebe (x, y, z) e retorna o vetor [P, Q, R].
        - superficie: Função que recebe (u, v) e retorna [x(u,v), y(u,v), z(u,v)].
        - u_limits, v_limits: Tuplas definindo o domínio paramétrico da superfície.
        - curva_bordo: Função que recebe t e retorna [x(t), y(t), z(t)] (o bordo de S).
        - t_limits: Tupla (t_inicio, t_fim) para a curva de bordo.

        Retorna:
        - Dicionário com 'integral_linha', 'integral_superficie' e 'diferenca_absoluta'.
        """
        t_inicio, t_fim = t_limits
        u_min, u_max = u_limits
        v_min, v_max = v_limits

        # --- LADO ESQUERDO: Integral de Linha no Bordo ---
        def integrando_linha(t):
            r = curva_bordo(t)
            x, y, z = r[0], r[1], r[2]
            dx_dt = self.derivadas.DerivadaCentralizada(lambda tau: curva_bordo(tau)[0], t)
            dy_dt = self.derivadas.DerivadaCentralizada(lambda tau: curva_bordo(tau)[1], t)
            dz_dt = self.derivadas.DerivadaCentralizada(lambda tau: curva_bordo(tau)[2], t)
            
            F_val = F(x, y, z)
            return F_val[0]*dx_dt + F_val[1]*dy_dt + F_val[2]*dz_dt
        
        lado_esquerdo = self._integrar_simpson_1d(integrando_linha, t_inicio, t_fim)

        # --- LADO DIREITO: Integral de Superfície do Rotacional ---
        def integrando_superficie(u, v):
            r = superficie(u, v)
            x, y, z = r[0], r[1], r[2]
            
            # Vetores tangentes r_u e r_v via derivada numérica
            r_u = np.array([
                self.derivadas.DerivadaCentralizada(lambda tau: superficie(tau, v)[0], u),
                self.derivadas.DerivadaCentralizada(lambda tau: superficie(tau, v)[1], u),
                self.derivadas.DerivadaCentralizada(lambda tau: superficie(tau, v)[2], u)
            ])
            r_v = np.array([
                self.derivadas.DerivadaCentralizada(lambda tau: superficie(u, tau)[0], v),
                self.derivadas.DerivadaCentralizada(lambda tau: superficie(u, tau)[1], v),
                self.derivadas.DerivadaCentralizada(lambda tau: superficie(u, tau)[2], v)
            ])
            
            # Vetor normal dS = (r_u × r_v) du dv
            normal = np.cross(r_u, r_v)
            
            # Rotacional de F no ponto (x,y,z) usando seu DerivadasSolver
            # Nota: DerivadaRotacional3D espera P, Q, R como funções separadas ou usa lambdas
            P_func = lambda x_val, y_val, z_val: F(x_val, y_val, z_val)[0]
            Q_func = lambda x_val, y_val, z_val: F(x_val, y_val, z_val)[1]
            R_func = lambda x_val, y_val, z_val: F(x_val, y_val, z_val)[2]
            
            rot_F = self.derivadas.DerivadaRotacional3D(P_func, Q_func, R_func, x, y, z)
            
            return np.dot(rot_F, normal)
        
        # Integração dupla aninhada
        def integrando_u(u):
            return self._integrar_simpson_1d(lambda v: integrando_superficie(u, v), v_min, v_max)
        
        lado_direito = self._integrar_simpson_1d(integrando_u, u_min, u_max)

        return {
            "teorema": "Stokes",
            "integral_linha": lado_esquerdo,
            "integral_superficie": lado_direito,
            "diferenca_absoluta": abs(lado_esquerdo - lado_direito)
        }

    def CalcularTeoremaDaDivergencia(self, F, volume_limits, superficies_limite):
        """
        Valida numericamente o Teorema da Divergência (Gauss):
        ∭_V (∇ · F) dV = ∬_{∂V} F · dS

        Parâmetros:
        - F: Função que recebe (x, y, z) e retorna o vetor [P, Q, R].
        - volume_limits: Tupla ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
        - superficies_limite: Lista de tuplas (superficie_func, u_limits, v_limits) 
          que compõem a superfície fechada ∂V.

        Retorna:
        - Dicionário com 'integral_volume', 'integral_superficie_fechada' e 'diferenca_absoluta'.
        """
        x_limits, y_limits, z_limits = volume_limits

        # --- LADO ESQUERDO: Integral de Volume da Divergência ---
        def integrando_volume(x, y, z):
            P_func = lambda x_val, y_val, z_val: F(x_val, y_val, z_val)[0]
            Q_func = lambda x_val, y_val, z_val: F(x_val, y_val, z_val)[1]
            R_func = lambda x_val, y_val, z_val: F(x_val, y_val, z_val)[2]
            
            # Reutiliza o método já existente no seu DerivadasSolver
            return self.derivadas.DerivadaDivergente3D(P_func, Q_func, R_func, x, y, z)
        
        lado_esquerdo = self._integrar_simpson_3d(integrando_volume, x_limits, y_limits, z_limits)

        # --- LADO DIREITO: Integral de Superfície Fechada (Fluxo) ---
        fluxo_total = 0.0
        for superficie, u_limits, v_limits in superficies_limite:
            u_min, u_max = u_limits
            v_min, v_max = v_limits
            
            def integrando_superficie(u, v):
                r = superficie(u, v)
                x, y, z = r[0], r[1], r[2]
                
                r_u = np.array([
                    self.derivadas.DerivadaCentralizada(lambda tau: superficie(tau, v)[0], u),
                    self.derivadas.DerivadaCentralizada(lambda tau: superficie(tau, v)[1], u),
                    self.derivadas.DerivadaCentralizada(lambda tau: superficie(tau, v)[2], u)
                ])
                r_v = np.array([
                    self.derivadas.DerivadaCentralizada(lambda tau: superficie(u, tau)[0], v),
                    self.derivadas.DerivadaCentralizada(lambda tau: superficie(u, tau)[1], v),
                    self.derivadas.DerivadaCentralizada(lambda tau: superficie(u, tau)[2], v)
                ])
                
                normal = np.cross(r_u, r_v)
                F_val = F(x, y, z)
                return np.dot(F_val, normal)
            
            def integrando_u(u):
                return self._integrar_simpson_1d(lambda v: integrando_superficie(u, v), v_min, v_max)
            
            fluxo_total += self._integrar_simpson_1d(integrando_u, u_min, u_max)
        
        lado_direito = fluxo_total

        return {
            "teorema": "Divergencia (Gauss)",
            "integral_volume": lado_esquerdo,
            "integral_superficie_fechada": lado_direito,
            "diferenca_absoluta": abs(lado_esquerdo - lado_direito)
        }