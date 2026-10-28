from equacoes.equacoes import EquacoesSolver
import math
import numpy as np

class FuncoesSolver():
    def __init__(self):
        """
        Solver de funcoes. as listadas sao:
        1. Funcoes polinomiais
        2. Funcoes logaritmicas
        3. Funcoes exponenciais
        4. Funcoes trigonométricas

        Convenção: os métodos devolvem a(s) RAIZ(ES) x de f(x) = 0
        (exceto Racional, que avalia P(x)/Q(x)).
        """

    def PolinomialGrau1(self, a, b, x=None):
        """
        Função polinomial de 1º grau:

            f(x) = a*x + b

        Parâmetros:
            a -> coeficiente de x
            b -> termo independente
            x -> valor no qual deseja calcular f(x)

        Se x for informado:
            retorna o valor de f(x).

        Se x não for informado:
            retorna a raiz da função, isto é, o valor de x
            para o qual f(x) = 0.
        """

        self.a = a
        self.b = b

        # Função constante
        if self.a == 0:

            if self.b == 0:
                if x is None:
                    return "Infinitas raízes: f(x) = 0 para todo x real."
                return "f(x) = 0 para qualquer valor de x."

            if x is None:
                return "A função é constante e não possui raiz."

            return f"f({x}) = {self.b}"

        # Se x foi fornecido, calcula f(x)
        if x is not None:
            resultado = self.a * x + self.b
            return resultado

        # Caso x não seja fornecido, calcula a raiz
        raiz = -self.b / self.a

        return raiz



    def PolinomialGrau2(self, a, b, c, x=None):
        """
        Função polinomial de 2º grau:

            f(x) = a*x² + b*x + c

        Parâmetros:
            a -> coeficiente de x²
            b -> coeficiente de x
            c -> termo independente
            x -> valor no qual deseja calcular f(x)

        Se x for informado:
            calcula f(x).

        Se x não for informado:
            calcula as raízes da equação.
        """

        self.a = a
        self.b = b
        self.c = c

        # Se a = 0, não é uma função de 2º grau.
        if self.a == 0:
           return self.PolinomialGrau1(self.b, self.c, x)

        # Calcula o valor da função
        if x is not None:
            resultado = (
                self.a * x**2
                + self.b * x
                + self.c
            )

            return resultado

        # Calcula o discriminante
        delta = self.b**2 - 4 * self.a * self.c

        # Sem raízes reais
        if delta < 0:
            return "A equação não possui raízes reais."

        # Uma raiz real
        if delta == 0:
            raiz = -self.b / (2 * self.a)

            return (
                raiz
            )

        # Duas raízes reais
        raiz1 = (-self.b + delta**0.5) / (2 * self.a)
        raiz2 = (-self.b - delta**0.5) / (2 * self.a)

        return (
            raiz1, raiz2
        )

    def cbrt(self, val):
        """Calcula a raiz cúbica real mantendo o sinal correto."""
        return math.copysign(abs(val) ** (1/3), val)

    def PolinominalGrau3(self, a, b, c, d):
        """
        Solução da equação ax^3 + bx^2 + cx + d = 0 pelo Método de Cardano-Tartaglia / Vieta.
        
        Nota sobre o coeficiente 'a':
        Não é necessário que a = 1. A fórmula normaliza a equação original 
        dividindo todos os coeficientes por 'a' (convertendo-a para a forma 
        reduzida t^3 + p*t + q = 0) e aplica o deslocamento (shift = b / (3 * a)) 
        para retornar as raízes corretas para qualquer 'a' diferente de zero.
        Se a = 0 a equação deixa de ser cúbica e cai para o grau 2.
        Só são devolvidas raízes REAIS (no caso delta > 0 há também um par complexo).
        """
        self.a, self.b, self.c, self.d = a, b, c, d

        if a == 0:
            return self.PolinomialGrau2(b, c, d)

        # Conversão para a forma reduzida t^3 + p*t + q = 0
        p = (3 * a * c - b ** 2) / (3 * a ** 2)
        q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
        
        delta = (q / 2) ** 2 + (p / 3) ** 3

        # Tolerância RELATIVA à escala dos termos de delta. Uma tolerância absoluta
        # (ex.: 1e-11) zera delta indevidamente quando as raízes são pequenas,
        # pois delta escala como (tamanho das raízes)^6.
        # 1e-14 foi escolhido empiricamente: detecta raízes duplas/triplas com coeficientes
        # inteiros e evita "forçar" raiz dupla em casos mal condicionados.
        escala = max((q / 2) ** 2, abs((p / 3) ** 3))
        if abs(delta) <= 1e-14 * escala:
            delta = 0.0

        shift = b / (3 * a)

        if delta > 0:
            sqrt_delta = math.sqrt(delta)
            # Soma o termo com o MESMO sinal de -q/2 (sem cancelamento subtrativo);
            # o outro cubo sai de u*v = -p/3.
            u = self.cbrt(-q / 2 + math.copysign(sqrt_delta, -q))
            v = -p / (3 * u)
            x = u + v - shift
            return x

        elif delta == 0:
            u = self.cbrt(-q / 2)
            x1 = 2 * u - shift          # raiz simples
            x2 = -u - shift             # raiz dupla (tripla se p = q = 0)
            return x1, x2

        else:
            # Quando delta < 0, existem 3 raízes reais distintas (Método Trigonométrico)
            r = 2 * math.sqrt(-p / 3)
            
            # cos(theta) = -q / (2 * sqrt(-p^3 / 27))
            arg = -q / (2 * math.sqrt(-(p ** 3) / 27))
            cos_arg = min(max(arg, -1.0), 1.0)  # Evita DomainError por arredondamento
            theta = math.acos(cos_arg)
            
            x1 = r * math.cos(theta / 3) - shift
            x2 = r * math.cos((theta + 2 * math.pi) / 3) - shift
            x3 = r * math.cos((theta + 4 * math.pi) / 3) - shift
            
            return x1, x2, x3

    def _raizes_quadratica(self, B, C, escala):
        """
        Raízes REAIS de y^2 + B*y + C = 0 (lista com 0, 1 ou 2 valores).
        Um discriminante ligeiramente negativo, dentro do erro de arredondamento
        (tolerância relativa à escala dos termos), é tratado como zero.
        """
        delta = B ** 2 - 4 * C
        ref = max(B ** 2, 4 * abs(C), escala)
        if delta < 0 and -delta <= 1e-12 * ref:
            delta = 0.0

        if delta < 0:
            return []
        if delta == 0:
            return [-B / 2]
        raiz = delta ** 0.5
        return [(-B + raiz) / 2, (-B - raiz) / 2]

    def PolinomialGrau4(self, a, b, c, d, e):
        """
        Solução da equação ax^4 + bx^3 + cx^2 + dx + e = 0 pelo Método de Ferrari.

        Nota sobre o coeficiente 'a':
        Não é necessário que a = 1. A equação é reduzida à forma
        y^4 + p*y^2 + q*y + r = 0 pelo deslocamento x = y - b / (4 * a).
        Se a = 0 a equação deixa de ser quártica e cai para o grau 3.

        Se q = 0 a equação é biquadrada e é resolvida direto em y^2.
        Caso contrário, usa-se a cúbica resolvente
            z^3 + 2p*z^2 + (p^2 - 4r)*z - q^2 = 0
        (resolvida por PolinominalGrau3; basta uma raiz real z > 0, e a maior
        raiz real sempre serve) para fatorar a quártica em dois trinômios:
            y^2 - sqrt(z)*y + (z + p)/2 + q / (2*sqrt(z)) = 0
            y^2 + sqrt(z)*y + (z + p)/2 - q / (2*sqrt(z)) = 0

        Só são devolvidas raízes REAIS e DISTINTAS, em ordem crescente
        (uma raiz como número, várias como tupla).
        Raízes múltiplas perdem precisão por natureza do problema (uma raiz dupla
        só é obtida com cerca de metade dos dígitos corretos).
        """
        self.a, self.b, self.c, self.d, self.e = a, b, c, d, e

        if a == 0:
            return self.PolinominalGrau3(b, c, d, e)

        # Conversão para a forma reduzida y^4 + p*y^2 + q*y + r = 0
        p = (8 * a * c - 3 * b ** 2) / (8 * a ** 2)
        q = (b ** 3 - 4 * a * b * c + 8 * a ** 2 * d) / (8 * a ** 3)
        r = (-3 * b ** 4 + 256 * a ** 3 * e - 64 * a ** 2 * b * d + 16 * a * b ** 2 * c) / (256 * a ** 4)
        shift = b / (4 * a)

        # Escala típica de y^2 (p, sqrt|r| e |q|^(2/3) têm a mesma unidade)
        escala = max(abs(p), math.sqrt(abs(r)), abs(q) ** (2 / 3))

        if abs(q) <= 1e-14 * escala ** 1.5:
            # Biquadrada: y^4 + p*y^2 + r = 0  ->  w = y^2
            raizes_y = []
            for w in self._raizes_quadratica(p, r, escala ** 2):
                if w < -1e-12 * escala:
                    continue
                w = max(w, 0.0)
                raizes_y += [w ** 0.5, -(w ** 0.5)]
        else:
            # Cúbica resolvente: pega a maior raiz real (sempre positiva se q != 0,
            # pois o valor da cúbica em z = 0 é -q^2 < 0)
            cubica = self.PolinominalGrau3(1, 2 * p, p ** 2 - 4 * r, -q ** 2)
            candidatos = cubica if isinstance(cubica, tuple) else (cubica,)
            z = max(candidatos)

            # Refinamento por Newton: mantém a precisão relativa quando z é pequeno
            for _ in range(3):
                g = ((z + 2 * p) * z + (p ** 2 - 4 * r)) * z - q ** 2
                dg = (3 * z + 4 * p) * z + (p ** 2 - 4 * r)
                if dg == 0:
                    break
                z -= g / dg

            if z <= 0:
                return "Sinto muito, falha numérica ao resolver a cúbica resolvente."

            sz = math.sqrt(z)
            s = (z + p) / 2
            t = q / (2 * sz)
            raizes_y = (
                self._raizes_quadratica(-sz, s + t, escala)
                + self._raizes_quadratica(sz, s - t, escala)
            )

        # Volta para x e remove duplicatas (raízes múltiplas aparecem repetidas)
        ref = max(math.sqrt(escala), abs(shift))
        distintas = []
        for x in sorted(y - shift for y in raizes_y):
            if not distintas or abs(x - distintas[-1]) > 1e-8 * max(ref, abs(x)):
                distintas.append(x)

        if not distintas:
            return "A equação não possui raízes reais."
        if len(distintas) == 1:
            return distintas[0]
        return tuple(distintas)

    def Exponencial(self, a, b, c, d, k):
        """
        Inicializa a função exponencial.
        Modelo: f(x) = a * b^(k*x + c) + d. Devolve a raiz de f(x) = 0:
            x = (ln(-d/a) - c*ln(b)) / (k*ln(b))
        Requer b > 0, b != 1, k != 0 e -d/a > 0 para existir solução real.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.k = k

        if self.b <= 0 or self.b == 1:
            return "Sinto muito, a base precisa satisfazer b > 0 e b != 1."
        if self.k == 0:
            return "Sinto muito, com k = 0 a função é constante e não tem raiz isolada."
        if self.a == 0:
            if self.d == 0:
                return "Infinitas soluções: com a = 0 e d = 0, f(x) = 0 para todo x real."
            return "Sem solução: com a = 0 e d != 0 a função é constante e não nula."

        alvo = -(self.d / self.a)
        if alvo <= 0:
            return f"Sinto muito, -d/a = {alvo} não é positivo: sem solução real."

        ln_b = math.log(self.b)
        resultado = (math.log(alvo) - self.c * ln_b) / (self.k * ln_b)
        return resultado

    def Logaritmica(self, a, b, c, d, k):
        """
        Inicializa a função logaritmica.
        Modelo: f(x) = a * log_b(k*x + c) + d. Devolve a raiz de f(x) = 0:
            x = (b^(-d/a) - c) / k
        Requer a != 0, k != 0, b > 0 e b != 1.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.k = k

        if self.a == 0 or self.k == 0:
            return "Sinto muito, é preciso a != 0 e k != 0."
        if self.b <= 0 or self.b == 1:
            return "Sinto muito, a base precisa satisfazer b > 0 e b != 1."

        x = (self.b ** (-(self.d / self.a)) - self.c) / self.k
        return x
    
    def ModuloAbsoluto(self, a, b, c):
        """
        Inicializa a função de módulo absoluto.
        Resolve |a*x + b| = c.
        Atente-se para o caso do valor de c ser menor que zero, pois nesse caso a função não terá solução real.
        """
        self.a = a
        self.b = b
        self.c = c
        
        if self.c < 0:
            return f"Sinto muito, o valor de  c = {self.c}, é menor que zero. Por tanto a operação nao tem solução real."

        if self.a == 0:
            if abs(self.b) == self.c:
                return "Infinitas soluções: com a = 0 e |b| = c a igualdade vale para todo x real."
            return "Sem solução: com a = 0 e |b| != c a igualdade nunca é satisfeita."

        x1 = (self.c - self.b) / self.a
        x2 = (-self.c - self.b) / self.a

        return x1, x2

    def Racional(self, P_x, Q_x):
        """
        Inicializa a função racional.
        Esta função requer que você ultilize funções auxiliares para definir P(x) e Q(x).
        Por exemplo, você pode instanciar uma do pacote de funções do proprio Omega e passa-las como parametros.
        (P_x e Q_x precisam ser NÚMEROS, i.e. P e Q já avaliados no ponto desejado.)
        """
        self.P_x = P_x
        self.Q_x = Q_x

        if self.Q_x == 0:
            return "Sinto muito, Q(x) = 0: a função racional não está definida nesse ponto."

        resultado = self.P_x / self.Q_x
        return resultado

    def _validar_trig(self, a, k, n):
        """Devolve uma mensagem de erro se não der para calcular o zero; senão, None."""
        if k == 0:
            return "Sinto muito, com k = 0 a função é constante e não tem raiz isolada."
        if a == 0:
            return "Sinto muito, com a = 0 a função é identicamente nula: todo x é raiz."
        if n != int(n):
            return f"Sinto muito, n = {n} precisa ser inteiro."
        return None
    
    def Seno(self, a, k, n, phi):
        """
        Inicializa a função trigonométrica do seno.
        Devolve o n-ésimo zero de f(x) = a*sin(k*x + phi), com n inteiro.
        """
        self.a = a
        self.k = k
        self.phi = phi
        self.n = n

        erro = self._validar_trig(self.a, self.k, self.n)
        if erro:
            return erro

        x = (self.n * math.pi - self.phi) / self.k

        return x


    def Cosseno(self, a, k, n, phi):
        """
        Inicializa a função trigonométrica do cosseno.
        Devolve o n-ésimo zero de f(x) = a*cos(k*x + phi), com n inteiro.
        """
        self.a = a
        self.k = k
        self.phi = phi
        self.n = n

        erro = self._validar_trig(self.a, self.k, self.n)
        if erro:
            return erro

        x = ((math.pi / 2) + self.n * math.pi - self.phi) / self.k

        return x

    def Tangente(self, a, k, n, phi):
        """
        Inicializa a função trigonométrica da tangente.
        Devolve o n-ésimo zero de f(x) = a*tan(k*x + phi), com n inteiro.
        """
        self.a = a
        self.k = k
        self.phi = phi
        self.n = n

        erro = self._validar_trig(self.a, self.k, self.n)
        if erro:
            return erro

        x = (self.n * math.pi - self.phi) / self.k

        return x

    def limite_numerico(self, f, x0, h=1e-6, direcao='ambos'):
        """
        Estima o limite de f(x) quando x -> x0 avaliando f perto de x0
        (aproximação numérica, não é cálculo simbólico).

        direcao: 'esquerda' (x -> x0 por baixo), 'direita' (x -> x0 por cima)
        ou 'ambos'. Em 'ambos', se os limites laterais diferem (ex.: 1/x ou
        sinal(x) em 0), o limite não existe e é devolvida uma mensagem em vez
        de uma média sem sentido.

        Limitações: a precisão é da ordem de h (ou pior, se f sofre cancelamento
        numérico perto de x0). Se f diverge para o MESMO infinito dos dois lados
        (ex.: 1/x² em 0), o resultado é um número enorme, que deve ser lido como infinito.
        """
        if direcao == 'esquerda':
            return f(x0 - h)
        if direcao == 'direita':
            return f(x0 + h)

        esquerda = f(x0 - h)
        direita = f(x0 + h)

        if not math.isclose(esquerda, direita, rel_tol=1e-3, abs_tol=1e-3):
            return (f"Sinto muito, os limites laterais são diferentes "
                    f"(esquerda ≈ {esquerda}, direita ≈ {direita}): o limite não existe.")

        return (esquerda + direita) / 2

    def HipoerbolicaSeno(self, x):
        return math.sinh(x)
    
    def HipoerbolicaCosseno(self, x):
        return math.cosh(x)

    def HipoerbolicaTangente(self, x):
        return math.tanh(x)

    def HipoerbolicaCotangente(self, x):
        if x == 0:
            raise ValueError("A cotangente hiperbolica nao esta definida em zero.")
        return 1.0 / math.tanh(x)

    def HipoerbolicaSecante(self, x):
        exponencial = math.exp(-abs(x))
        return 2.0 * exponencial / (1.0 + exponencial**2)

    def HiperbolicaCossecante(self, x):
        if x == 0:
            raise ValueError("A cossecante hiperbolica nao esta definida em zero.")
        modulo = abs(x)
        exponencial = math.exp(-modulo)
        valor = 2.0 * exponencial / (-math.expm1(-2.0 * modulo))
        return math.copysign(valor, x)

    def MetodoLagrange(self, pontos_x, pontos_y, x_desejado):
        """Calcula o valor interpolado em x_desejado usando o Polinômio de Lagrange.

        pontos_x: lista com as coordenadas x dos pontos (ex: [1, 2, 4])
        pontos_y: lista com as coordenadas y correspondentes (ex: [2, 1, 5])
        x_desejado: valor de x onde você quer estimar y
        """
        n = len(pontos_x)
        p_x = 0.0  # Acumulador do valor final interpolado

        for i in range(n):
            # Termo L_i(x) começa em 1
            l_i = 1.0
            for j in range(n):
                if i != j:
                    l_i *= (x_desejado - pontos_x[j]) / (
                        pontos_x[i] - pontos_x[j]
                    )

            # Soma y_i * L_i(x) ao resultado total
            p_x += pontos_y[i] * l_i

        return p_x


class FuncoesSolverNaoLinear():
    def __init__(self):
        pass

    def ResolverSistemaNaoLinear(
        self,
        residuo,
        chute_inicial,
        tolerancia=1e-10,
        max_iter=100,
        passo_jacobiano=1e-5,
        amortecimento_inicial=1e-3,
        escalas_residuos=None,
    ):
        """Resolve F(x)=0 para sistemas acoplados por Newton amortecido.

        residuo(x) deve retornar um vetor numerico finito. O numero de equacoes
        pode diferir do numero de incognitas; em sistemas sobredeterminados, a
        etapa resolve o problema linearizado de minimos quadrados. escalas_residuos
        opcional normaliza equacoes com unidades/magnitudes diferentes.

        Retorna solucao, residuos, norma, estado de convergencia e diagnosticos.
        Convergencia e local e depende do chute inicial; nao garante unicidade.
        """
        if not callable(residuo):
            raise TypeError("residuo deve ser uma funcao que recebe o vetor de incognitas.")
        x = np.asarray(chute_inicial, dtype=float)
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("chute_inicial deve ser um vetor nao vazio e finito.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter deve ser um inteiro positivo.")

        tolerancia = float(tolerancia)
        passo_jacobiano = float(passo_jacobiano)
        amortecimento = float(amortecimento_inicial)
        if not all(math.isfinite(v) and v > 0.0 for v in (tolerancia, passo_jacobiano, amortecimento)):
            raise ValueError("tolerancia, passo_jacobiano e amortecimento devem ser positivos e finitos.")

        def avaliar(incognitas):
            valores = np.asarray(residuo(incognitas.copy()), dtype=float).reshape(-1)
            if valores.size == 0 or not np.all(np.isfinite(valores)):
                raise ValueError("residuo deve retornar valores finitos e nao vazios.")
            return valores

        bruto = avaliar(x)
        if escalas_residuos is None:
            escalas = np.ones(bruto.size, dtype=float)
        else:
            escalas = np.asarray(escalas_residuos, dtype=float)
            if escalas.shape != bruto.shape or not np.all(np.isfinite(escalas)) or np.any(escalas <= 0.0):
                raise ValueError("escalas_residuos deve ser um vetor positivo com o shape dos residuos.")

        def avaliar_escalado(incognitas):
            return avaliar(incognitas) / escalas

        valores = bruto / escalas
        norma = float(np.linalg.norm(valores))
        convergiu = norma <= tolerancia
        mensagem = "tolerancia de residuos atingida" if convergiu else "maximo de iteracoes nao atingido"
        iteracoes = aceitos = rejeitados = 0
        passo_final = math.inf

        while not convergiu and iteracoes < max_iter:
            jacobiano = np.empty((valores.size, x.size), dtype=float)
            for coluna in range(x.size):
                h = passo_jacobiano * max(1.0, abs(x[coluna]))
                def amostrar(multiplicador):
                    perturbado = x.copy()
                    perturbado[coluna] += multiplicador * h
                    try:
                        return avaliar_escalado(perturbado)
                    except (ArithmeticError, ValueError, FloatingPointError):
                        return None

                menos_dois, menos_um, mais_um, mais_dois = (
                    amostrar(multiplicador)
                    for multiplicador in (-2.0, -1.0, 1.0, 2.0)
                )
                if all(
                    amostra is not None
                    for amostra in (menos_dois, menos_um, mais_um, mais_dois)
                ):
                    jacobiano[:, coluna] = (
                        menos_dois - 8.0 * menos_um + 8.0 * mais_um - mais_dois
                    ) / (12.0 * h)
                    continue

                amostras_direita = [amostrar(multiplicador) for multiplicador in (1.0, 2.0, 3.0, 4.0)]
                if all(amostra is not None for amostra in amostras_direita):
                    jacobiano[:, coluna] = (
                        -25.0 * valores
                        + 48.0 * amostras_direita[0]
                        - 36.0 * amostras_direita[1]
                        + 16.0 * amostras_direita[2]
                        - 3.0 * amostras_direita[3]
                    ) / (12.0 * h)
                    continue

                amostras_esquerda = [amostrar(-multiplicador) for multiplicador in (1.0, 2.0, 3.0, 4.0)]
                if all(amostra is not None for amostra in amostras_esquerda):
                    jacobiano[:, coluna] = (
                        25.0 * valores
                        - 48.0 * amostras_esquerda[0]
                        + 36.0 * amostras_esquerda[1]
                        - 16.0 * amostras_esquerda[2]
                        + 3.0 * amostras_esquerda[3]
                    ) / (12.0 * h)
                    continue

                raise ValueError(
                    f"Nao foi possivel calcular o Jacobiano no dominio do parametro {coluna}."
                )

            coluna_normas = np.maximum(np.linalg.norm(jacobiano, axis=0), 1.0)
            aceitou = False
            tentativas = min(14, max_iter - iteracoes)
            for _ in range(tentativas):
                regularizador = np.diag(np.sqrt(amortecimento) * coluna_normas)
                matriz_aumentada = np.vstack((jacobiano, regularizador))
                rhs_aumentado = np.concatenate((-valores, np.zeros(x.size)))
                incremento = np.linalg.lstsq(matriz_aumentada, rhs_aumentado, rcond=None)[0]
                passo_final = float(np.linalg.norm(incremento))
                candidato = x + incremento
                try:
                    bruto_candidato = avaliar(candidato)
                    valores_candidato = bruto_candidato / escalas
                    norma_candidata = float(np.linalg.norm(valores_candidato))
                except (ArithmeticError, ValueError, FloatingPointError):
                    norma_candidata = math.inf

                iteracoes += 1
                if norma_candidata < norma:
                    x = candidato
                    bruto = bruto_candidato
                    valores = valores_candidato
                    norma = norma_candidata
                    amortecimento = max(amortecimento / 3.0, 1e-14)
                    aceitos += 1
                    aceitou = True
                    convergiu = norma <= tolerancia
                    if convergiu:
                        mensagem = "tolerancia de residuos atingida"
                    break

                amortecimento *= 10.0
                rejeitados += 1
                if amortecimento > 1e16:
                    break

            if not aceitou:
                mensagem = (
                    "maximo de iteracoes atingido"
                    if iteracoes >= max_iter
                    else "nenhum passo reduziu a norma dos residuos"
                )
                break
            if passo_final <= tolerancia * (1.0 + np.linalg.norm(x)) and not convergiu:
                mensagem = "incremento pequeno antes de atingir tolerancia de residuos"
                break

        if not convergiu and iteracoes >= max_iter:
            mensagem = "maximo de iteracoes atingido"
        return {
            "solucao": x,
            "residuos": bruto,
            "residuos_escalados": valores,
            "norma_residuo": norma,
            "convergiu": convergiu,
            "iteracoes": iteracoes,
            "passos_aceitos": aceitos,
            "passos_rejeitados": rejeitados,
            "passo_final": passo_final,
            "mensagem": mensagem,
            "limitacoes": [
                "Metodo local: depende do chute inicial e nao garante solucao unica.",
                "Raizes em que o Jacobiano perde posto podem exigir reformulacao ou regularizacao.",
                "O Jacobiano e aproximado por diferencas finitas de cinco pontos.",
            ],
        }

    def BuscarRaizesMultiplas(
        self,
        residuo,
        chutes_iniciais,
        tolerancia=1e-10,
        tolerancia_raizes=1e-7,
        max_iter=100,
        **opcoes_solver,
    ):
        """Executa o solver nao linear a partir de varios chutes e deduplica raizes.

        Retorna todas as raizes distintas encontradas nos chutes fornecidos,
        resultados por chute e `completude_garantida=False`: nenhum conjunto
        finito de chutes prova que todas as raizes de um problema geral foram
        encontradas.
        """
        if not callable(residuo):
            raise TypeError("residuo deve ser uma funcao.")
        chutes = list(chutes_iniciais)
        if not chutes:
            raise ValueError("Forneca ao menos um chute inicial.")
        tolerancia_raizes = float(tolerancia_raizes)
        if not math.isfinite(tolerancia_raizes) or tolerancia_raizes <= 0.0:
            raise ValueError("tolerancia_raizes deve ser positiva e finita.")

        resultados = []
        raizes = []
        for chute in chutes:
            resultado = self.ResolverSistemaNaoLinear(
                residuo,
                chute,
                tolerancia=tolerancia,
                max_iter=max_iter,
                **opcoes_solver,
            )
            resultados.append(resultado)
            if not resultado["convergiu"]:
                continue
            raiz = resultado["solucao"]
            if not any(
                np.linalg.norm(raiz - existente)
                <= tolerancia_raizes * (1.0 + np.linalg.norm(existente))
                for existente in raizes
            ):
                raizes.append(raiz.copy())

        return {
            "raizes_encontradas": raizes,
            "resultados_por_chute": resultados,
            "chutes_tentados": len(chutes),
            "completude_garantida": False,
            "aviso": "A busca multichute encontra raizes acessiveis aos chutes; nao certifica que todas foram encontradas.",
        }

    def NewtonRaphson(self, f, f_prime, x0, tol=1e-7, max_iter=100):
        """
        Método de Newton-Raphson para encontrar raízes de funções não lineares.
        
        :param f: Função para a qual queremos encontrar a raiz.
        :param f_prime: Derivada da função f.
        :param x0: Estimativa inicial da raiz.
        :param tol: Tolerância para o critério de convergência.
        :param max_iter: Número máximo de iterações.
        :return: Aproximação da raiz ou mensagem de falha.
        Esta função requer que você ultilize funções auxiliares para definir f e f_prime.
        Por exemplo, você pode instanciar uma do pacote de funções do proprio Omega e passa-las como parametros.
        """
        x = x0
        for i in range(max_iter):
            fx = f(x)
            fpx = f_prime(x)
            if fpx == 0:
                return "Derivada zero. Não é possível continuar."
            x_new = x - fx / fpx
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        return "Número máximo de iterações atingido. A raiz pode não ter sido encontrada."
    
    def MetodoSecantes(self, f, x0, x1, tol=1e-7, max_iter=100):
        """
        Método das Secantes para encontrar raízes de funções não lineares.
        
        :param f: Função para a qual queremos encontrar a raiz.
        :param x0: Primeira estimativa inicial da raiz.
        :param x1: Segunda estimativa inicial da raiz.
        :param tol: Tolerância para o critério de convergência.
        :param max_iter: Número máximo de iterações.
        :return: Aproximação da raiz ou mensagem de falha.
        Esta função requer que você ultilize funções auxiliares para definir f.
        Por exemplo, você pode instanciar uma do pacote de funções do proprio Omega e passa-las como parametros.
        """
        for i in range(max_iter):
            fx0 = f(x0)
            fx1 = f(x1)
            if fx1 - fx0 == 0:
                return "Divisão por zero. Não é possível continuar."
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            if abs(x_new - x1) < tol:
                return x_new
            x0, x1 = x1, x_new
        return "Número máximo de iterações atingido. A raiz pode não ter sido encontrada."

    def PolinomioGrauArbitrario(self, x, *coffs):
        """
        Por polinomios completos entendemos que podem ser representados\n
        Pela funcao soma de P(x), Sendo ai diferente de zero; isto é, nenhum coeficiente é nulo e, portanto, o polinômio possuirá todos os termos relativos às potências de x.\n
        Uma implementação possível do método de Horner.
        Parametros:\n

        x: O ponto para qual calcula-se o polinomio arbritario.
        coffs: conjunto de coeficientes, ex: 1, -4, 3, -2, 1, -9

        ONS: esta função tambem pode ser utilizada para numeros complexos por exemplp:\n
        P(x) = 1*x^2 - 4*x + 5  (coeficientes: 1, -4, 5)
        onde x:
        x_complexo = 1 + 2j
        """
        resutl = 0.0
        coeficientes = list(coffs)
        for i in range(len(coeficientes)):
            resutl = coeficientes[i] + resutl * x
        return resutl

    def MetodoBissecao(self, f, a, b, N_inter, Tol):
        """
        Metodo da bisseção.
        """
        lista = []
        
        # Opcional: verificar se há mudança de sinal inicial (Teorema de Bolzano)
        if f(a) * f(b) >= 0:
            raise ValueError("A função deve ter sinais opostos nas extremidades 'a' e 'b'.")

        for i in range(N_inter):
            # 1. Ponto médio correto (entre as posições a e b)
            c = (a + b) / 2
            lista.append(c)
            
            # 2. Critério de parada por tolerância do tamanho do intervalo
            if ((b - a) / 2) < Tol or f(c) == 0:
                break
            
            # 3. Atualização dos intervalos baseada no sinal de f(c)
            if f(a) * f(c) < 0:
                b = c  # A raiz está na metade esquerda
            else:
                a = c  # A raiz está na metade direita
                
        return lista

    def MetodoFalsaPosicao(self, f, x0, x1, N_inter, tol):
        k0 = f(x0)
        k1 = f(x1)
        lista = []

        if (f(x0) * f(x1)) > 0:
            return ValueError("x0 e x1 devem necessariamente ter sinais diferentes!")
        
        for i in range(N_inter):
            x = x1 - (k1 * (x1 - x0)) / (k1 - k0)
            lista.append(x)

            if abs((x - x1)) < tol:
                break

            k = f(x)
            if (k * k1) < 0:
                x0 = x
                k0 = k
            else:
                x1 = x
                k1 = k 
        return lista

    def MetodoInterpolacaoQuadInversa(self, f, x0, x1, x2, N_inter, tol):
        """Encontra a raiz usando Interpolação Quadrática Inversa.

        Precisa de 3 pontos iniciais: x0, x1, x2.
        """
        lista = []

        for i in range(N_inter):
            y0 = f(x0)
            y1 = f(x1)
            y2 = f(x2)

            # Se o valor da função for menor que a tolerância, encontramos a raiz
            if abs(y2) < tol:
                lista.append(x2)
                break

            # Evitar divisão por zero caso os valores de y sejam muito próximos
            if (y0 - y1) == 0 or (y0 - y2) == 0 or (y1 - y2) == 0:
                raise ValueError(
                    "Divisão por zero devido a valores de função iguais."
                )

            # Fórmula da Interpolação Quadrática Inversa (avaliada em y = 0)
            termo0 = (x0 * y1 * y2) / ((y0 - y1) * (y0 - y2))
            termo1 = (x1 * y0 * y2) / ((y1 - y0) * (y1 - y2))
            termo2 = (x2 * y0 * y1) / ((y2 - y0) * (y2 - y1))

            x3 = termo0 + termo1 + termo2
            lista.append(x3)

            # Critério de parada baseado na diferença entre as aproximações
            if abs(x3 - x2) < tol:
                break

            # Atualiza os três pontos para a próxima iteração (desliza a janela)
            x0 = x1
            x1 = x2
            x2 = x3

        return lista

    def MetodoBrent(self, f, a, b, tol=1e-7, max_iter=100):
        """Método de Brent para encontrar raízes de funções não lineares.

        Combina bisseção, secante e interpolação quadrática inversa de forma
        híbrida e segura.
        """
        fa = f(a)
        fb = f(b)

        # Validação inicial do Teorema de Bolzano
        if fa * fb >= 0:
            raise ValueError(
                "A função deve ter sinais opostos nas extremidades 'a' e 'b'."
            )

        # Garante que |f(b)| seja menor ou igual a |f(a)|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        c = a
        fc = fa
        mflag = True
        d = 0.0

        for _ in range(max_iter):
            fa = f(a)
            fb = f(b)
            fc = f(c)

            # Critério de parada
            if fb == 0 or abs(b - a) < tol:
                return b

            # Tenta interpolação quadrática inversa ou método das secantes
            if fa != fc and fb != fc:
                # Interpolação Quadrática Inversa (IQI)
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                    (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                    (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                # Método das Secantes
                s = b - fb * (b - a) / (fb - fa)

            # Condições complexas do Brent para decidir se aceita 's' ou usa bisseção
            cond1 = not ((3 * a + b) / 4 <= s <= b) if a < b else not (b <= s <= (3 * a + b) / 4)
            cond2 = mflag and (abs(s - b) >= abs(b - c) / 2)
            cond3 = (not mflag) and (abs(s - b) >= abs(c - d) / 2)
            cond4 = mflag and (abs(b - c) < tol)
            cond5 = (not mflag) and (abs(c - d) < tol)

            if cond1 or cond2 or cond3 or cond4 or cond5:
                # Recorre à bisseção se a interpolação não for segura
                s = (a + b) / 2
                mflag = True
            else:
                mflag = False

            fs = f(s)
            d = c
            c = b
            fc = fb

            if fa * fs < 0:
                b = s
            else:
                a = s

            # Ordena novamente os pontos
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa

        return b