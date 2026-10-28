import math
import numpy as np
from constantes.constantes import Constantes
import numbers

c = Constantes()
class IntegralSolverIndefinidas:
    def __init__(self, n_passos=1000):
        """
        Solver de integrais definidas.
        

        3. Exponencial
        4. Trigonométrico (seno e cosseno)
        5. Fração parcial linear

        Cada método devolve UMA primitiva F(x) + C, isto é, um número.
        C é a constante de integração: para ver a família de primitivas,
        faça um loop externo em torno de C.

        EXEMPLO DE USO:

            solve = IntegralSolverPrimitivas()
            for C in range(10):
                print(solve.RegraDaPotencia(2, 3, C))
                # print(solve.Logaritmico(2, C))
                # print(solve.Exponencial(2, 3, C))
                # print(solve.TrigonometricaSeno(2, 3, C))
                # print(solve.TrigonometricaCosseno(2, 3, C))
                # print(solve.FracaoParcialLinear(5, 2, 1, 3, C))
        """
        if not isinstance(n_passos, int) or n_passos <= 0:
            raise ValueError("n_passos deve ser um inteiro positivo.")
        # Força ser par para a regra de Simpson composta funcionar corretamente
        self.n_passos = n_passos if n_passos % 2 == 0 else n_passos + 1

    def RegraDaPotencia(self, x, n, C):
        """
        Integral pela regra da potência:

            ∫ x^n dx = x^(n+1) / (n+1) + C,   para n != -1

        Para n = -1 use Logaritmico (∫ 1/x dx = ln|x| + C).
        """
        self.x = x
        self.n = n
        self.C = C

        if self.n == -1:
            return "Sinto muito, a regra da potência não vale para n = -1. Use Logaritmico: ∫ 1/x dx = ln|x| + C."
        if self.x == 0 and self.n + 1 < 0:
            return "Sinto muito, x^(n+1) não está definido em x = 0 para n < -1."

        resultado = self.x ** (self.n + 1) / (self.n + 1) + self.C

        if isinstance(resultado, complex):
            return "Sinto muito, x negativo com expoente fracionário sai do domínio real."

        return resultado

    def Logaritmico(self, x, C):
        """
        Integral logarítmica:

            ∫ 1/x dx = ln|x| + C,   para x != 0

        O módulo |x| é necessário: a primitiva vale também para x < 0.
        """
        self.x = x
        self.C = C

        if self.x == 0:
            return "Sinto muito, 1/x não está definido em x = 0."

        resultado = math.log(abs(self.x)) + self.C
        return resultado
    
    def ExponencialBaseGenerica(self, a, x, C):
        """
        Integral exponencial com base genérica:

            ∫ a^x dx = a^x / ln(a) + C,   para a > 0 e a != 1
        """
        self.a = a
        self.x = x
        self.C = C

        if self.a <= 0 or self.a == 1:
            return "Sinto muito, a base a deve ser positiva e diferente de 1."

        resultado = self.a ** self.x / math.log(self.a) + self.C
        return resultado

    def LogaritimoNatural(self, x, C):
        """
        Integral do logaritmo natural:

            ∫ ln(x) dx = x * ln(x) - x + C,   para x > 0
        """
        self.x = x
        self.C = C

        if self.x <= 0:
            return "Sinto muito, ln(x) não está definido para x <= 0."

        resultado = self.x * math.log(self.x) - self.x + self.C
        return resultado

    def Exponencial(self, a, x, C):
        """
        Integral exponencial:

            ∫ e^(a*x) dx = e^(a*x) / a + C,   para a != 0

        Para a = 0 o integrando é 1 e a primitiva é x + C.
        """
        self.a = a
        self.x = x
        self.C = C

        if self.a == 0:
            return self.x + self.C

        resultado = math.exp(self.a * self.x) / self.a + self.C
        return resultado

    def TrigonometricaSeno(self, a, x, C):
        """
        Integral trigonométrica do seno:

            ∫ sin(a*x) dx = -cos(a*x) / a + C,   para a != 0

        Para a = 0 o integrando é 0 e a primitiva é apenas C.
        """
        self.a = a
        self.x = x
        self.C = C

        if self.a == 0:
            return self.C

        resultado = -math.cos(self.a * self.x) / self.a + self.C
        return resultado

    def TrigonometricaCosseno(self, a, x, C):
        """
        Integral trigonométrica do cosseno:

            ∫ cos(a*x) dx = sin(a*x) / a + C,   para a != 0

        Para a = 0 o integrando é 1 e a primitiva é x + C.
        """
        self.a = a
        self.x = x
        self.C = C

        if self.a == 0:
            return self.x + self.C

        resultado = math.sin(self.a * self.x) / self.a + self.C
        return resultado

    def TrigonometricaTangente(self, x, C):
        """
        Integral trigonométrica da tangente:

            ∫ tan(x) dx = -ln|cos(x)| + C,   para x != π/2 + kπ, k ∈ Z
         """
        self.x = x
        self.C = C

        if math.isclose(math.cos(self.x), 0, abs_tol=1e-9):
            return "Sinto muito, tan(x) não está definida em x = π/2 + kπ, k ∈ Z."

        resultado = -math.log(abs(math.cos(self.x))) + self.C
        return resultado
    
    def CotangenteQuadrado(self, x, C):
        """
        Integral da cotangente ao quadrado:
        
            ∫ cot²(x) dx = -cot(x) - x + C,   para x != kπ, k ∈ Z
        """
        self.x = x
        self.C = C

        if math.isclose(math.sin(self.x), 0, abs_tol=1e-9):
            return "Sinto muito, cot(x) não está definida em x = kπ, k ∈ Z."

        resultado = (-1 / math.tan(self.x)) - self.x + self.C
        return resultado

    def SecanteTangente(self, x, C):
        """
        Integral da secante vezes a tangente:
        
            ∫ sec(x) * tan(x) dx = sec(x) + C,   para x != π/2 + kπ, k ∈ Z
        """
        self.x = x
        self.C = C

        if math.isclose(math.cos(self.x), 0, abs_tol=1e-9):
            return "Sinto muito, sec(x) não está definida em x = π/2 + kπ, k ∈ Z."

        resultado = (1 / math.cos(self.x)) + self.C
        return resultado

    def CossecanteCotangente(self, x, C):
        """
        Integral da cossecante vezes a cotangente:
        
            ∫ csc(x) * cot(x) dx = -csc(x) + C,   para x != kπ, k ∈ Z
        """
        self.x = x
        self.C = C

        if math.isclose(math.sin(self.x), 0, abs_tol=1e-9):
            return "Sinto muito, csc(x) não está definida em x = kπ, k ∈ Z."

        resultado = -(1 / math.sin(self.x)) + self.C
        return resultado

    #Funções Inversas / Formas Racionais e Radicais

    def ArcoSenoPadrao(self, x, C):
        """
        Integral do arco seno:

            ∫ arcsin(x) dx = x * arcsin(x) + sqrt(1 - x²) + C,   para |x| <= 1
        """
        self.x = x
        self.C = C

        if abs(self.x) > 1:
            return "Sinto muito, arcsin(x) não está definido para |x| > 1."

        resultado = self.x * math.asin(self.x) + math.sqrt(1 - self.x ** 2) + self.C
        return resultado

    def ArcosenoTangente(self, x, C):
        """
        Integral do arcoseno tangente

        ∫ arctan(x) dx = x * arctan(x) - (1/2) * ln(1 + x²) + C,   para todo x real
        """
        self.x = x
        self.C = C

        resultado = self.x * math.atan(self.x) - (1/2) * math.log(1 + self.x ** 2) + self.C
        return resultado
    
    def RegraDoArcoTangenteGeral(self, x, a, C):
        """
        Regra do Arco Tangente Geral

        (1.0 / a) * atan(x / a) + C
        """
        self.a = a
        self.x = x
        self.C = C

        if self.a == 0:
            return "Sinto muito, 'a' não pode ser zero."

        modulo_a = abs(self.a)
        resultado = math.atan(self.x / modulo_a) / modulo_a + self.C
        return resultado

    def RegraArcoSenoGeral(self, a, x, C):
        """
        Regra do Arco Seno Geral

        asin(x / a)
        """
        self.a = a
        self.x = x
        self.C = C

        if self.a == 0:
            return "Sinto muito, 'a' não pode ser zero."
        if abs(self.x) > abs(self.a):
            return "Sinto muito, é preciso |x| <= |a| para o arco seno estar definido."

        resultado = math.asin(self.x / abs(self.a)) + self.C
        return resultado
    
    def FracaoParcialLinear(self, A, a, b, x, C):
        """
        Integral de uma fração com denominador linear e numerador constante:

            ∫ A / (a*x + b) dx = (A / a) * ln|a*x + b| + C,   para a != 0

        Para a = 0 o integrando é a constante A / b e a primitiva é (A / b) * x + C.
        Não é definida em a*x + b = 0 (polo da função).
        """
        self.A = A
        self.a = a
        self.b = b
        self.x = x
        self.C = C

        if self.a == 0:
            if self.b == 0:
                return "Sinto muito, com a = 0 e b = 0 o denominador é zero."
            return (self.A / self.b) * self.x + self.C

        denominador = self.a * self.x + self.b
        if denominador == 0:
            return "Sinto muito, a*x + b = 0: a função não está definida nesse ponto."

        resultado = (self.A / self.a) * math.log(abs(denominador)) + self.C
        return resultado

    #Integrais de Funções Hiperbólicas

    def SenoHiperbolico(self, x, C):
        """
        Seno Hiperbólico

        cosh(x)
        """
        self.x = x
        self.C = C

        resultado = math.cosh(self.x) + self.C
        return resultado

    def CossenoHiperbolico(self, x, C):
        self.x = x
        self.C = C

        resultado = math.sinh(self.x) + self.C
        return resultado

    def TangenteHiperbolica(self, x, C):
        self.x = x
        self.C = C

        resultado = math.log(math.cosh(self.x)) + self.C
        return resultado

    def SecanteHiperbolicaQuadrado(self, x, C):
        self.x = x
        self.C = C

        resultado = math.tanh(self.x) + self.C
        return resultado

    def IntegralPolinomialGrau1(self, a, b, x, C):
        """
        Calcula a integral indefinida de um polinômio de 1º grau: f(x) = a*x + b
        Primitiva: (a/2)*x² + b*x
        O usuário insere apenas os coeficientes: a, b e o valor de x.
        """
        termo1 = a * self.RegraDaPotencia(x, 1, 0)  # Para o termo 'a*x' (expoente 1)
        termo2 = b * x                               # Para o termo constante 'b'
        self.C = C
        return termo1 + termo2 + self.C

    def IntegralPolinomialGrau2(self, a, b, c, x, C):
        """
        Calcula a integral indefinida de um polinômio de 2º grau: f(x) = a*x² + b*x + c
        Primitiva: (a/3)*x³ + (b/2)*x² + c*x
        O usuário insere apenas os coeficientes: a, b, c e o valor de x.
        """
        termo1 = a * self.RegraDaPotencia(x, 2, 0)  # Para o termo 'a*x²' (expoente 2)
        termo2 = b * self.RegraDaPotencia(x, 1, 0)  # Para o termo 'b*x' (expoente 1)
        termo3 = c * x                               # Para o termo constante 'c'
        self.C = C
        return termo1 + termo2 + termo3 + self.C

    def IntegralPolinomialGrau3(self, a, b, c, d, x, C):
        """
        Calcula a integral indefinida de um polinômio de 3º grau: f(x) = a*x³ + b*x² + c*x + d
        Primitiva: (a/4)*x⁴ + (b/3)*x³ + (c/2)*x² + d*x
        O usuário insere apenas os coeficientes: a, b, c, d e o valor de x.
        """
        termo1 = a * self.RegraDaPotencia(x, 3, 0)  # Para o termo 'a*x³' (expoente 3)
        termo2 = b * self.RegraDaPotencia(x, 2, 0)  # Para o termo 'b*x²' (expoente 2)
        termo3 = c * self.RegraDaPotencia(x, 1, 0)  # Para o termo 'c*x' (expoente 1)
        termo4 = d * x                               # Para o termo constante 'd'
        self.C = C
        return termo1 + termo2 + termo3 + termo4 + self.C

    def IntegralPolinomialGrau4(self, a, b, c, d, e, x, C):
        """Calcula a integral indefinida de um polinômio de 4º grau:

        f(x) = a*x⁴ + b*x³ + c*x² + d*x + e
        Primitiva: (a/5)*x⁵ + (b/4)*x⁴ + (c/3)*x³ + (d/2)*x² + e*x
        O usuário insere apenas os coeficientes: a, b, c, d, e e o valor de x.
        """
        termo1 = a * self.RegraDaPotencia(x, 4, 0)  # Para o termo 'a*x⁴' (expoente 4)
        termo2 = b * self.RegraDaPotencia(x, 3, 0)  # Para o termo 'b*x³' (expoente 3)
        termo3 = c * self.RegraDaPotencia(x, 2, 0)  # Para o termo 'c*x²' (expoente 2)
        termo4 = d * self.RegraDaPotencia(x, 1, 0)  # Para o termo 'd*x'  (expoente 1)
        termo5 = e * x                               # Para o termo constante 'e'

        self.C = C

        return termo1 + termo2 + termo3 + termo4 + termo5 + self.C

    def IntegralExponencialCompostaQuadratica(self, A, k, x, C):
        """Calcula a integral do tipo: ∫ A * x * e^(k * x²) dx.

        Primitiva: (A / (2*k)) * e^(k * x²)
        """
        self.C = C
        if k == 0:
            return "Sinto muito, 'k' não pode ser zero."

        return (A / (2 * k)) * math.exp(k * (x**2)) + self.C


    def _integrar_simpson_1d(self, f, a, b):
        """
        Regra de Simpson composta para integração 1D (Erro de truncamento O(h^4)).
        """
        n = self.n_passos
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([float(f(xi)) for xi in x])
        
        # Fórmula de Simpson: (h/3) * [y0 + 4*(ímpares) + 2*(pares) + yn]
        soma = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
        return (h / 3.0) * soma

    def IntegralDeLinha(self, campo_vetorial, curva, t_inicio, t_fim):
        """
        Calcula a integral de linha de um campo vetorial ao longo de uma curva parametrizada.
        Fórmula: ∫_C F · dr = ∫_a^b F(r(t)) · r'(t) dt
        (Usada para calcular trabalho de forças, circulação, etc.)

        Parâmetros:
        - campo_vetorial: callable que recebe (x, y, z) e retorna um array [Fx, Fy, Fz].
        - curva: callable que recebe t e retorna um array [x(t), y(t), z(t)].
        - t_inicio, t_fim: limites de integração do parâmetro t.

        Retorna:
        - Valor escalar (float) da integral de linha.
        """
        from derivadas import DerivadasSolver
        d = DerivadasSolver()
        
        def integrando(t):
            # 1. Posição no ponto t
            r = np.array(curva(t), dtype=float)
            
            # 2. Derivada numérica da curva r'(t) = [x'(t), y'(t), z'(t)]
            # Reutilizamos seu DerivadasSolver para manter a consistência do pacote
            dr_dt = np.array([
                d.DerivadaCentralizada(lambda tau: curva(tau)[0], t),
                d.DerivadaCentralizada(lambda tau: curva(tau)[1], t),
                d.DerivadaCentralizada(lambda tau: curva(tau)[2], t)
            ])
            
            # 3. Campo vetorial avaliado na posição r
            F = np.array(campo_vetorial(r[0], r[1], r[2]), dtype=float)
            
            # 4. Produto escalar F · r'(t)
            return np.dot(F, dr_dt)
        
        return self._integrar_simpson_1d(integrando, t_inicio, t_fim)

    def IntegralDeSuperficie(self, campo_vetorial, superficie, u_limits, v_limits):
        """
        Calcula a integral de superfície de um campo vetorial sobre uma superfície parametrizada.
        Fórmula: ∬_S F · dS = ∬_D F(r(u,v)) · (r_u × r_v) du dv
        (Usada para calcular fluxo de campos, como fluxo elétrico ou de fluidos).

        Parâmetros:
        - campo_vetorial: callable que recebe (x, y, z) e retorna [Fx, Fy, Fz].
        - superficie: callable que recebe (u, v) e retorna [x(u,v), y(u,v), z(u,v)].
        - u_limits: tupla (u_min, u_max).
        - v_limits: tupla (v_min, v_max).

        Retorna:
        - Valor escalar (float) da integral de superfície (fluxo).
        """
        from derivadas import DerivadasSolver
        d = DerivadasSolver()
        u_min, u_max = u_limits
        v_min, v_max = v_limits
        
        def integrando(u, v):
            r = np.array(superficie(u, v), dtype=float)
            
            # Derivada parcial em relação a u: r_u
            ru = np.array([
                d.DerivadaCentralizada(lambda tau: superficie(tau, v)[0], u),
                d.DerivadaCentralizada(lambda tau: superficie(tau, v)[1], u),
                d.DerivadaCentralizada(lambda tau: superficie(tau, v)[2], u)
            ])
            
            # Derivada parcial em relação a v: r_v
            rv = np.array([
                d.DerivadaCentralizada(lambda tau: superficie(u, tau)[0], v),
                d.DerivadaCentralizada(lambda tau: superficie(u, tau)[1], v),
                d.DerivadaCentralizada(lambda tau: superficie(u, tau)[2], v)
            ])
            
            # Vetor normal à superfície: r_u × r_v
            normal = np.cross(ru, rv)
            
            # Campo vetorial avaliado na superfície
            F = np.array(campo_vetorial(r[0], r[1], r[2]), dtype=float)
            
            # Produto escalar F · (r_u × r_v)
            return np.dot(F, normal)
        
        # Integração dupla aninhada usando Simpson composto
        def integrando_u(u):
            return self._integrar_simpson_1d(lambda v: integrando(u, v), v_min, v_max)
        
        return self._integrar_simpson_1d(integrando_u, u_min, u_max)

    def IntegralDeVolume(self, campo_escalar, x_limits, y_limits, z_limits):
        """
        Calcula a integral de volume (tripla) de um campo escalar em uma região retangular 3D.
        Fórmula: ∭_V f(x,y,z) dV = ∫ ∫ ∫ f(x,y,z) dx dy dz
        (Usada para calcular massa, carga total, volume geométrico, etc.)

        Parâmetros:
        - campo_escalar: callable que recebe (x, y, z) e retorna um escalar (float).
        - x_limits: tupla (x_min, x_max).
        - y_limits: tupla (y_min, y_max).
        - z_limits: tupla (z_min, z_max).

        Retorna:
        - Valor escalar (float) da integral de volume.
        """
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        z_min, z_max = z_limits
        
        # Integração tripla aninhada
        def integrando_x(x):
            def integrando_y(y):
                return self._integrar_simpson_1d(lambda z: campo_escalar(x, y, z), z_min, z_max)
            return self._integrar_simpson_1d(integrando_y, y_min, y_max)
        
        return self._integrar_simpson_1d(integrando_x, x_min, x_max)

class IntegralSolverDefinidas:
    def __init__(self):
        """
        Solver de expressões de Integrais Indefinidas (Fórmula da Primitiva F(x)).
        
        Cada método retorna o valor da expressão primitiva F(x) para um dado x,
        omitindo a constante de integração C.
        """

        self.mu_0 = c.MU_0
    def Integral(self, x, n):
        """Primitiva de x**n avaliada em x, sem constante de integracao."""
        if n == -1:
            return self.IntegralDoReciproco(x)
        return self.RegraDaPotencia(x, n)
    # --- 1. Funções Potência e Polinomiais ---

    def RegraDaConstante(self, k, x):
        """
        ∫ k dx = k * x
        """
        return k * x

    def RegraDaPotencia(self, x, n):
        """
        ∫ x^n dx = x^(n+1) / (n+1),  para n != -1
        """
        if n == -1:
            return "Sinto muito, a regra da potência não vale para n = -1. Use IntegralDoReciproco."
        if x == 0 and n + 1 < 0:
            return "Sinto muito, x^(n+1) não está definido em x = 0 para n < -1."

        resultado = (x ** (n + 1)) / (n + 1)
        if isinstance(resultado, complex):
            return "Sinto muito, x negativo com expoente fracionário sai do domínio real."

        return resultado

    def IntegralDoReciproco(self, x):
        """
        ∫ 1/x dx = ln|x|,  para x != 0
        """
        if x == 0:
            return "Sinto muito, 1/x não está definido em x = 0."

        return math.log(abs(x))

    # --- 2. Funções Exponenciais e Logarítmicas ---

    def ExponencialNatural(self, x):
        """
        ∫ e^x dx = e^x
        """
        return math.exp(x)

    def ExponencialBaseGenerica(self, a, x):
        """
        ∫ a^x dx = a^x / ln(a),  para a > 0 e a != 1
        """
        if a <= 0 or a == 1:
            return "Sinto muito, a base 'a' deve ser positiva e diferente de 1."

        return (a ** x) / math.log(a)

    def IntegralDoLogaritmoNatural(self, x):
        """
        ∫ ln(x) dx = x * ln(x) - x,  para x > 0
        """
        if x <= 0:
            return "Sinto muito, ln(x) não está definido para x <= 0."

        return x * math.log(x) - x

    # --- 3. Funções Trigonométricas Básicas ---

    def IntegralDoSeno(self, x, a=1):
        """
        ∫ sin(a*x) dx = -cos(a*x) / a,  para a != 0
        """
        if a == 0:
            return 0.0

        return -math.cos(a * x) / a

    def IntegralDoCosseno(self, x, a=1):
        """
        ∫ cos(a*x) dx = sin(a*x) / a,  para a != 0
        """
        if a == 0:
            return x

        return math.sin(a * x) / a

    def IntegralDaTangente(self, x):
        """
        ∫ tan(x) dx = -ln|cos(x)|
        """
        if math.isclose(math.cos(x), 0, abs_tol=1e-9):
            return "Sinto muito, tan(x) não está definida em x = π/2 + kπ."

        return -math.log(abs(math.cos(x)))

    def IntegralDaSecanteQuadrado(self, x):
        """
        ∫ sec²(x) dx = tan(x)
        """
        if math.isclose(math.cos(x), 0, abs_tol=1e-9):
            return "Sinto muito, sec²(x) não está definida em x = π/2 + kπ."

        return math.tan(x)

    def IntegralDaCossecanteQuadrado(self, x):
        """
        ∫ csc²(x) dx = -cot(x)
        """
        if math.isclose(math.sin(x), 0, abs_tol=1e-9):
            return "Sinto muito, csc²(x) não está definida em x = kπ."

        return -1.0 / math.tan(x)

    def IntegralSecanteVezesTangente(self, x):
        """
        ∫ sec(x)*tan(x) dx = sec(x)
        """
        if math.isclose(math.cos(x), 0, abs_tol=1e-9):
            return "Sinto muito, sec(x)*tan(x) não está definida em x = π/2 + kπ."

        return 1.0 / math.cos(x)

    def IntegralCossecanteVezesCotangente(self, x):
        """
        ∫ csc(x)*cot(x) dx = -csc(x)
        """
        if math.isclose(math.sin(x), 0, abs_tol=1e-9):
            return "Sinto muito, csc(x)*cot(x) não está definida em x = kπ."

        return -1.0 / math.sin(x)

    # --- 4. Funções Inversas / Formas Racionais e Radicais ---

    def IntegralDoArcoSenoPadrao(self, x):
        """
        ∫ 1 / sqrt(1 - x²) dx = arcsin(x)
        """
        if abs(x) >= 1:
            return "Sinto muito, 1/sqrt(1 - x²) não está definido para |x| >= 1."

        return math.asin(x)

    def IntegralDoArcoTangentePadrao(self, x):
        """
        ∫ 1 / (1 + x²) dx = arctan(x)
        """
        return math.atan(x)

    def RegraDoArcoTangenteGeral(self, x, a):
        """
        ∫ 1 / (a² + x²) dx = (1/a) * arctan(x/a)
        """
        if a == 0:
            return "Sinto muito, 'a' não pode ser zero."

        modulo_a = abs(a)
        return math.atan(x / modulo_a) / modulo_a

    def RegraDoArcoSenoGeral(self, x, a):
        """
        ∫ 1 / sqrt(a² - x²) dx = arcsin(x/a)
        """
        if a == 0:
            return "Sinto muito, 'a' não pode ser zero."
        if abs(x) > abs(a):
            return "Sinto muito, e preciso |x| <= |a|."

        return math.asin(x / abs(a))

    # --- 5. Fração Parcial Linear ---

    def FracaoParcialLinear(self, A, a, b, x):
        """
        ∫ A / (a*x + b) dx = (A / a) * ln|a*x + b|
        """
        if a == 0:
            if b == 0:
                return "Sinto muito, com a = 0 e b = 0 o denominador é zero."
            return (A / b) * x

        denominador = a * x + b
        if denominador == 0:
            return "Sinto muito, a*x + b = 0: a função não está definida nesse ponto."

        return (A / a) * math.log(abs(denominador))

    # --- 6. Funções Hiperbólicas ---

    def SenoHiperbolico(self, x):
        """
        ∫ sinh(x) dx = cosh(x)
        """
        return math.cosh(x)

    def CossenoHiperbolico(self, x):
        """
        ∫ cosh(x) dx = sinh(x)
        """
        return math.sinh(x)

    def TangenteHiperbolica(self, x):
        """
        ∫ tanh(x) dx = ln(cosh(x))
        """
        return math.log(math.cosh(x))

    def SecanteHiperbolicaQuadrado(self, x):
        """
        ∫ sech²(x) dx = tanh(x)
        """
        return math.tanh(x)
    
# --- Polinômios (Integrais Indefinidas / Primitivas) ---

    def IntegralPolinomialGrau1(self, a, b, x):
        """
        Calcula a integral indefinida de um polinômio de 1º grau: f(x) = a*x + b
        Primitiva: (a/2)*x² + b*x
        O usuário insere apenas os coeficientes: a, b e o valor de x.
        """
        termo1 = a * self.RegraDaPotencia(x, 1)  # Para o termo 'a*x' (expoente 1)
        termo2 = self.RegraDaConstante(b, x)      # Para o termo constante 'b'
        
        return termo1 + termo2

    def IntegralPolinomialGrau2(self, a, b, c, x):
        """
        Calcula a integral indefinida de um polinômio de 2º grau: f(x) = a*x² + b*x + c
        Primitiva: (a/3)*x³ + (b/2)*x² + c*x
        O usuário insere apenas os coeficientes: a, b, c e o valor de x.
        """
        termo1 = a * self.RegraDaPotencia(x, 2)  # Para o termo 'a*x²' (expoente 2)
        termo2 = b * self.RegraDaPotencia(x, 1)  # Para o termo 'b*x' (expoente 1)
        termo3 = self.RegraDaConstante(c, x)      # Para o termo constante 'c'
        
        return termo1 + termo2 + termo3

    def IntegralPolinomialGrau3(self, a, b, c, d, x):
        """
        Calcula a integral indefinida de um polinômio de 3º grau: f(x) = a*x³ + b*x² + c*x + d
        Primitiva: (a/4)*x⁴ + (b/3)*x³ + (c/2)*x² + d*x
        O usuário insere apenas os coeficientes: a, b, c, d e o valor de x.
        """
        termo1 = a * self.RegraDaPotencia(x, 3)  # Para o termo 'a*x³' (expoente 3)
        termo2 = b * self.RegraDaPotencia(x, 2)  # Para o termo 'b*x²' (expoente 2)
        termo3 = c * self.RegraDaPotencia(x, 1)  # Para o termo 'c*x' (expoente 1)
        termo4 = self.RegraDaConstante(d, x)      # Para o termo constante 'd'
        
        return termo1 + termo2 + termo3 + termo4
    
    def IntegralPolinomialGrau4(self, a, b, c, d, e, x):
        """Calcula a integral indefinida de um polinômio de 4º grau:

        f(x) = a*x⁴ + b*x³ + c*x² + d*x + e
        Primitiva: (a/5)*x⁵ + (b/4)*x⁴ + (c/3)*x³ + (d/2)*x² + e*x
        O usuário insere apenas os coeficientes: a, b, c, d, e e o valor de x.
        """
        termo1 = a * self.RegraDaPotencia(x, 4)  # Para o termo 'a*x⁴' (expoente 4)
        termo2 = b * self.RegraDaPotencia(x, 3)  # Para o termo 'b*x³' (expoente 3)
        termo3 = c * self.RegraDaPotencia(x, 2)  # Para o termo 'c*x²' (expoente 2)
        termo4 = d * self.RegraDaPotencia(x, 1)  # Para o termo 'd*x'  (expoente 1)
        termo5 = self.RegraDaConstante(e, x)  # Para o termo constante 'e'

        return termo1 + termo2 + termo3 + termo4 + termo5

    def IntegralExponencialCompostaQuadratica(self, A, k, x):
        """Calcula a integral do tipo: ∫ A * x * e^(k * x²) dx.

        Primitiva: (A / (2*k)) * e^(k * x²)
        """
        if k == 0:
            return "Sinto muito, 'k' não pode ser zero."

        return (A / (2 * k)) * math.exp(k * (x**2))


    def TransformadaLaplace(self, n, s):
        """Transformada de Laplace de t**n, n inteiro >= 0, s > 0: n!/s**(n+1)."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n deve ser um inteiro nao negativo.")
        s = float(s)
        if not math.isfinite(s) or s <= 0.0:
            raise ValueError("s deve ser positivo e finito para a transformada de t**n.")
        from equacoes.equacoes import EquacoesSolver
        eq = EquacoesSolver()
        return eq.fatorial(n) / s ** (n + 1)


    def calcular_biot_savart_2d(self, ponto_obs, x_vals, y_vals, K_func):
        """
        Calcula o campo magnético B (vetor 3D) gerado por uma distribuição 
        superficial de corrente K(x, y) no plano xy (z=0).
        
        Parâmetros:
        - ponto_obs: array ou tupla [x, y, z] do ponto onde se mede o campo.
        - x_vals, y_vals: arrays 1D com a malha da fonte de corrente.
        - K_func: função que retorna o vetor densidade de corrente K = [Kx, Ky, Kz] dado (x, y).
        """
        dx = x_vals[1] - x_vals[0]
        dy = y_vals[1] - y_vals[0]
        dA = dx * dy  # Elemento de área infinitesimal
        
        r_obs = np.array(ponto_obs, dtype=float)
        Bx, By, Bz = 0.0, 0.0, 0.0
        
        # Duplo loop para integrar sobre a superfície 2D
        for xp in x_vals:
            for yp in y_vals:
                r_fonte = np.array([xp, yp, 0.0], dtype=float)
                R_vetor = r_obs - r_fonte
                R_dist = np.linalg.norm(R_vetor)
                
                # Evita divisão por zero se o ponto de observação estiver em cima da fonte
                if R_dist == 0:
                    continue
                    
                K = np.array(K_func(xp, yp), dtype=float)
                produto_vet = np.cross(K, R_vetor)
                
                # Contribuição infinitesimal dB
                d_B = (self.mu_0 / (4 * np.pi)) * produto_vet / (R_dist ** 3) * dA
                
                Bx += d_B[0]
                By += d_B[1]
                Bz += d_B[2]
                
        return np.array([Bx, By, Bz])

    def calcular_biot_savart_3d(self, ponto_obs, x_vals, y_vals, z_vals, J_func):
        """
        Calcula o campo magnético B (vetor 3D) gerado por uma distribuição 
        volumétrica de corrente J(x, y, z) em um espaço 3D.
        
        Parâmetros:
        - ponto_obs: array ou tupla [x, y, z] do ponto onde se mede o campo.
        - x_vals, y_vals, z_vals: arrays 1D com a malha volumétrica da fonte.
        - J_func: função que retorna o vetor densidade de corrente volumétrica J = [Jx, Jy, Jz] dado (x, y, z).
        """
        dx = x_vals[1] - x_vals[0]
        dy = y_vals[1] - y_vals[0]
        dz = z_vals[1] - z_vals[0]
        dV = dx * dy * dz  # Elemento de volume infinitesimal
        
        r_obs = np.array(ponto_obs, dtype=float)
        Bx, By, Bz = 0.0, 0.0, 0.0
        
        # Triplo loop para integrar sobre o volume 3D
        for xp in x_vals:
            for yp in y_vals:
                for zp in z_vals:
                    r_fonte = np.array([xp, yp, zp], dtype=float)
                    R_vetor = r_obs - r_fonte
                    R_dist = np.linalg.norm(R_vetor)
                    
                    if R_dist == 0:
                        continue
                        
                    J = np.array(J_func(xp, yp, zp), dtype=float)
                    produto_vet = np.cross(J, R_vetor)
                    
                    # Contribuição infinitesimal dB para volume
                    d_B = (self.mu_0 / (4 * np.pi)) * produto_vet / (R_dist ** 3) * dV
                    
                    Bx += d_B[0]
                    By += d_B[1]
                    Bz += d_B[2]
                    
        return np.array([Bx, By, Bz])

    def RegraDeSimpson(self, f, a, b, N=100):
        """Integral composta de Simpson 1/3 de f em [a, b].

        N e o numero par de subintervalos. Para estimativa de erro controlada,
        use IntegracaoSimpsonAdaptativa.
        """
        if not callable(f):
            raise TypeError("f deve ser uma funcao numerica.")
        if isinstance(N, bool) or not isinstance(N, int) or N < 2 or N % 2:
            raise ValueError("N deve ser um inteiro par >= 2.")
        a, b = float(a), float(b)
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Os limites devem ser finitos.")
        h = (b - a) / N
        soma = f(a) + f(b)
        for i in range(1, N):
            valor = f(a + i * h)
            soma += (4.0 if i % 2 else 2.0) * valor
        return h * soma / 3.0

    def IntegracaoSimpsonAdaptativa(self, f, a, b, tolerancia=1e-9, profundidade_maxima=20):
        """Simpson adaptativo com estimativa do erro local e Richardson.

        Retorna integral, erro_estimado, numero de avaliacoes e indicador de
        convergencia. A estimativa pressupoe f suficientemente suave em cada
        subintervalo; descontinuidades devem ser separadas em pontos de corte.
        """
        if not callable(f):
            raise TypeError("f deve ser uma funcao numerica.")
        a, b = float(a), float(b)
        tolerancia = float(tolerancia)
        if not all(math.isfinite(valor) for valor in (a, b, tolerancia)):
            raise ValueError("Limites e tolerancia devem ser finitos.")
        if tolerancia <= 0.0:
            raise ValueError("tolerancia deve ser positiva.")
        if (
            isinstance(profundidade_maxima, bool)
            or not isinstance(profundidade_maxima, int)
            or profundidade_maxima < 1
        ):
            raise ValueError("profundidade_maxima deve ser um inteiro >= 1.")
        if a == b:
            return {
                "integral": 0.0,
                "erro_estimado": 0.0,
                "avaliacoes": 0,
                "convergiu": True,
            }
        sinal = 1.0 if b > a else -1.0
        esquerda, direita = (a, b) if b > a else (b, a)
        cache = {}

        def avaliar(x):
            if x not in cache:
                valor = np.asarray(f(x), dtype=float)
                if valor.ndim != 0 or not np.isfinite(valor):
                    raise ValueError("f deve retornar um escalar real finito.")
                cache[x] = float(valor)
            return cache[x]

        def simpson(x0, x1, f0, fm, f1):
            return (x1 - x0) * (f0 + 4.0 * fm + f1) / 6.0

        x0, x1 = esquerda, direita
        xm = 0.5 * (x0 + x1)
        f0, fm, f1 = avaliar(x0), avaliar(xm), avaliar(x1)
        inicial = simpson(x0, x1, f0, fm, f1)

        def refinar(x0, x1, f0, fm, f1, integral, tol, nivel):
            xm = 0.5 * (x0 + x1)
            xl = 0.5 * (x0 + xm)
            xr = 0.5 * (xm + x1)
            fl, fr = avaliar(xl), avaliar(xr)
            esquerda_local = simpson(x0, xm, f0, fl, fm)
            direita_local = simpson(xm, x1, fm, fr, f1)
            soma = esquerda_local + direita_local
            erro = abs(soma - integral) / 15.0
            if erro <= tol:
                return soma + (soma - integral) / 15.0, erro, True
            if nivel == 0:
                return soma, erro, False
            res_esq = refinar(
                x0, xm, f0, fl, fm, esquerda_local, tol / 2.0, nivel - 1
            )
            res_dir = refinar(
                xm, x1, fm, fr, f1, direita_local, tol / 2.0, nivel - 1
            )
            return (
                res_esq[0] + res_dir[0],
                res_esq[1] + res_dir[1],
                res_esq[2] and res_dir[2],
            )

        valor, erro, convergiu = refinar(
            x0, x1, f0, fm, f1, inicial, tolerancia, profundidade_maxima
        )
        return {
            "integral": sinal * valor,
            "erro_estimado": erro,
            "avaliacoes": len(cache),
            "convergiu": convergiu,
        }

    def RegraDoTrapezioSimples(self, b, a, f):
        h = b - a
        I = (h/2) * (f(a) + f(b))

        return I


    def RegraDoTrapezioComposta(self, a, b, f, max_inter, tol):
        lista = []
        
        # O loop 'n' controla a quantidade de subintervalos a cada tentativa
        for n in range(1, max_inter + 1):
            h = (b - a) / n
            
            # Soma começa com os extremos (peso 1) divididos por 2
            soma = 0.5 * (f(a) + f(b))
            
            # Soma todos os pontos internos x_i
            for i in range(1, n):
                x_i = a + i * h
                soma += f(x_i)
                
            # Multiplica pelo passo h para achar o valor da integral (I)
            I = h * soma
            lista.append(I)
            
            # Critério de parada opcional baseado na tolerância
            # (Geralmente compara-se com a iteração anterior, mas mantendo sua lógica:)
            if abs(I) <= tol:
                break
                
        return lista