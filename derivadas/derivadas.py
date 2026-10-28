import math
import numbers
import numpy as np

class DerivadasSolver():
    """
    Derivadas do Omega.

    O usuário fornece FUNÇÕES (callables: algo que recebe x e devolve f(x),
    como as do pacote de funções do Omega) e o ponto x. As derivadas de f e g
    são calculadas aqui dentro; o usuário nunca precisa informar f' ou g'.

    Limitação: as derivadas de f e g são numéricas (diferença central),
    ou seja, aproximações com erro da ordem de h², não derivadas exatas.
    """

    def __init__(self, h=1e-5):
        if (
            isinstance(h, bool)
            or not isinstance(h, numbers.Real)
            or not math.isfinite(float(h))
            or h <= 0
        ):
            raise ValueError("h deve ser positivo e finito.")
        self.h = float(h)

    @staticmethod
    def _derivada_cinco_pontos(f, x, h):
        return (
            f(x - 2.0 * h)
            - 8.0 * f(x - h)
            + 8.0 * f(x + h)
            - f(x + 2.0 * h)
        ) / (12.0 * h)

    def Derivada(self, f, x):
        """Derivada central de cinco pontos, com erro de truncamento O(h^4)."""
        return self._derivada_cinco_pontos(f, x, self.h)

    def DerivadaAdaptativa(
        self, f, x, tolerancia=1e-10, max_refinamentos=12
    ):
        """Estimativa adaptativa por Richardson da derivada primeira.

        Retorna um dicionario com valor, erro_estimado, passo e convergiu.
        O erro e uma estimativa por comparacao de malhas, nao um limite rigoroso.
        """
        tolerancia = float(tolerancia)
        if not math.isfinite(tolerancia) or tolerancia <= 0.0:
            raise ValueError("tolerancia deve ser positiva e finita.")
        if (
            isinstance(max_refinamentos, bool)
            or not isinstance(max_refinamentos, int)
            or max_refinamentos < 1
        ):
            raise ValueError("max_refinamentos deve ser um inteiro positivo.")

        passo = self.h
        anterior = self._derivada_cinco_pontos(f, x, passo)
        erro = math.inf
        valor = anterior
        for _ in range(max_refinamentos):
            passo *= 0.5
            atual = self._derivada_cinco_pontos(f, x, passo)
            diferenca = atual - anterior
            erro = abs(diferenca) / 15.0
            valor = atual + diferenca / 15.0
            if erro <= tolerancia * max(1.0, abs(valor)):
                return {
                    "valor": valor,
                    "erro_estimado": erro,
                    "passo": passo,
                    "convergiu": True,
                }
            anterior = atual
        return {
            "valor": valor,
            "erro_estimado": erro,
            "passo": passo,
            "convergiu": False,
        }

    def QuocienteIncremental(self, f, x, a):
        """[f(x) - f(a)] / (x - a): inclinação da secante; a derivada é o limite quando x -> a."""
        if x == a:
            raise ValueError("x e a não podem ser iguais (divisão por zero).")
        return (f(x) - f(a)) / (x - a)

    def DerivadaConstante(self, c=0):
        """d/dx (c) = 0"""
        return 0.0

    def DerivadaRegraDaPotencia(self, n, x, c=1):
        """d/dx (c * x^n) = c * n * x^(n-1)  (fórmula exata, sem aproximação)"""
        if n == 0:
            return 0.0
        if x == 0 and n < 1:
            raise ZeroDivisionError("Derivada de x^n com n < 1 não existe em x = 0.")
        resultado = c * n * x ** (n - 1)
        if isinstance(resultado, complex):
            raise ValueError("x negativo com expoente fracionário sai do domínio real.")
        return resultado

    def DerivadaSomaSubtracao(self, f, g, x, operacao="soma"):
        """(f ± g)'(x) = f'(x) ± g'(x)"""
        if operacao == "soma":
            return self.Derivada(f, x) + self.Derivada(g, x)
        if operacao == "subtracao":
            return self.Derivada(f, x) - self.Derivada(g, x)
        raise ValueError("operacao deve ser 'soma' ou 'subtracao'.")

    def DerivadaProduto(self, f, g, x):
        """(f * g)'(x) = f'(x) * g(x) + f(x) * g'(x)"""
        return self.Derivada(f, x) * g(x) + f(x) * self.Derivada(g, x)

    def DerivadaQuociente(self, f, g, x):
        """(f / g)'(x) = (f'(x) * g(x) - f(x) * g'(x)) / g(x)^2, com g(x) != 0"""
        gx = g(x)
        if gx == 0:
            raise ZeroDivisionError("g(x) = 0: o quociente não está definido.")
        return (self.Derivada(f, x) * gx - f(x) * self.Derivada(g, x)) / gx ** 2

    def DerivadaRegraDaCadeia(self, f, g, x):
        """(f∘g)'(x) = f'(g(x)) * g'(x). f é a função de FORA, g a de DENTRO."""
        return self.Derivada(f, g(x)) * self.Derivada(g, x)

    def DerivadaProgressiva(self, f, x):
        """Derivada progressiva de cinco pontos, com erro de truncamento O(h^4)."""
        h = self.h
        return (
            -25.0 * f(x)
            + 48.0 * f(x + h)
            - 36.0 * f(x + 2.0 * h)
            + 16.0 * f(x + 3.0 * h)
            - 3.0 * f(x + 4.0 * h)
        ) / (12.0 * h)

    def DerivadaCentralizada(self, f, x):
        """Alias da diferenca central de cinco pontos usada por Derivada."""
        return self.Derivada(f, x)

    def SegundaDerivadaCentralizada(self, f, x):
        """Segunda derivada central de cinco pontos, com erro O(h^4)."""
        h = self.h
        return (
            -f(x + 2.0 * h)
            + 16.0 * f(x + h)
            - 30.0 * f(x)
            + 16.0 * f(x - h)
            - f(x - 2.0 * h)
        ) / (12.0 * h**2)

    def LaplacianoSegundaOrdem(self, f, *coords):
        """
        Laplaciano numérico de f no ponto dado: soma das segundas derivadas
        parciais, uma por coordenada.

            ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² + ...

        f      : função das coordenadas espaciais, f(x, y, z).
        coords : o ponto (x, y, z) onde calcular.
        """
        total = 0.0
        for i in range(len(coords)):
            def g(s, i=i):
                ponto = list(coords)
                ponto[i] = s
                return f(*ponto)
            total += self.SegundaDerivadaCentralizada(g, coords[i])
        return total

    def LaplacianoPrimeiraOrdem(self, f, *coords):
        """
        Laplaciano numérico de f no ponto dado: soma das segundas derivadas
        parciais, uma por coordenada.

            ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² + ...

        f      : função das coordenadas espaciais, f(x, y, z).
        coords : o ponto (x, y, z) onde calcular.
        """
        total = 0.0
        for i in range(len(coords)):
            def g(s, i=i):
                ponto = list(coords)
                ponto[i] = s
                return f(*ponto)
            total += self.DerivadaCentralizada(g, coords[i])
        return total

    def DerivadaMista(self, f, x, y):
        """
        Calcula a derivada parcial mista ∂²f/∂x∂y numericamente 
        aplicando a derivada em relação a y sobre a função derivada em relação a x.
        """
        # Derivada parcial em relação a x mantendo y constante como função auxiliar
        def f_x(y_val):
            # Retorna a derivada em x avaliada em (x_fixo, y_val)
            def sub_f(x_val):
                return f(x_val, y_val)
            return self.Derivada(sub_f, x)
            
        return self.Derivada(f_x, y)
    
    def DerivadaSegundaX(self, f, x, y):
        """Calcula ∂²f/∂x² mantendo y constante."""
        # Cria uma função de uma variável apenas para o eixo x
        def f_de_x(x_val):
            return f(x_val, y)
        
        # Aplica a segunda derivada central em cima dessa função unidimensional
        return self.SegundaDerivadaCentralizada(f_de_x, x)

    def DerivadaSegundaY(self, f, x, y):
        """Calcula ∂²f/∂y² mantendo x constante."""
        # Cria uma função de uma variável apenas para o eixo y
        def f_de_y(y_val):
            return f(x, y_val)
        
        # Aplica a segunda derivada central em cima dessa função unidimensional
        return self.SegundaDerivadaCentralizada(f_de_y, y)

    def DerivadaTerceiraX(self, f, x, y):
        def derivadaSegunda(x):
            return self.DerivadaSegundaX(f,x,y)

        return self.DerivadaCentralizada(derivadaSegunda, x)

    def Derivada2x1y(self, f, x, y):
        """Calcula a derivada mista ∂³f / ∂x²∂y"""     
        # Mais direto usando a DerivadaCentralizada variando y:
        # Como a derivada centralizada precisa de uma função que varia y, fazemos:
        def func_para_y(y_val):
            return self.DerivadaSegundaX(f, x, y_val)
            
        return self.DerivadaCentralizada(func_para_y, y)

    def Derivada1x2y(self, f, x, y):
        def f_para_x(x):
            return self.DerivadaSegundaY(f, x, y)
        return self.DerivadaCentralizada(f_para_x, x)
    
    def DerivadaQuarta4x(self, f, x, y):
        """Calcula ∂⁴f / ∂x⁴"""
        def func_x(x_val):
            return self.DerivadaSegundaX(f, x_val, y)
        return self.SegundaDerivadaCentralizada(func_x, x)

    def Derivada3x1y(self, f, x, y):
        """Calcula ∂⁴f / ∂x³∂y"""
        def func_y(y_val):
            # Derivada terceira em x (que varia x, mantendo y_val fixo)
            # Ou você pode fazer uma função auxiliar que encapsula a 3x
            def func_x(x_val):
                return self.DerivadaSegundaX(f, x_val, y_val)
            return self.DerivadaCentralizada(func_x, x)
        
        return self.DerivadaCentralizada(func_y, y)

    def Derivada2x2y(self, f, x, y):
        """Calcula ∂⁴f / ∂x²∂y²"""
        def func_y(y_val):
            return self.DerivadaSegundaX(f, x, y_val)
        return self.SegundaDerivadaCentralizada(func_y, y)

    def Derivada1x3y(self, f, x, y):
        """Calcula ∂⁴f / ∂x∂y³"""
        def func_x(x_val):
            # Aqui você usaria a lógica da 3ª ordem em y
            def func_y(y_val):
                return self.DerivadaSegundaY(f, x_val, y_val)
            return self.DerivadaCentralizada(func_y, y)
            
        return self.DerivadaCentralizada(func_x, x)

    def DerivadaQuarta4y(self, f, x, y):
        """Calcula ∂⁴f / ∂y⁴"""
        def func_y(y_val):
            return self.DerivadaSegundaY(f, x, y_val)
        return self.SegundaDerivadaCentralizada(func_y, y)


    def GradienteMatrizJacobiana(self, funcoes, x_vetor):
        import tensorflow as tf
        
        """
        A matriz Jacobiana é uma representacao generalizada\n
        de uma matriz de dimensões m x n que organiza todas as\n
        derivadas parciais de primeira ordem de um sistema de funções.\n
        seja par auma função vetoria F(x) = [f_1(x), f_2(x)..., f_m(x)]\n
        calculamos o gradiente de uma unica fita para n funcoes passadas no parametro *f.\n

    
        Calcula a Matriz Jacobiana (m x n) real usando diferenciação automática.
        
        Parâmetros:
        - funcoes: Lista ou tupla de callables [f1, f2, ..., fm]
        - x_vetor: Vetor/array com os pontos de entrada [x1, x2, ..., xn]
        """
        # Convertemos o ponto x para um tensor que o TF possa monitorar
        x_tensor = tf.Variable(x_vetor, dtype=tf.float64)
        
        jacobiana = []
        
        with tf.GradientTape(persistent=True) as tape:
            # Avaliamos cada função do sistema no ponto x
            # Cada f(x) gera uma saída escalar que será derivada em relação a x_tensor
            resultados = [f(x_tensor) for f in funcoes]
            
        for y in resultados:
            # Calcula o gradiente de cada função 'y' em relação a todo o vetor 'x'
            # Isso nos dá uma linha da matriz Jacobiana
            gradiente_linha = tape.gradient(y, x_tensor)
            jacobiana.append(gradiente_linha.numpy())
            
        del tape
        return jacobiana.numpy()

    def GradienteMatrizHessiana(funcao, x_vetor):
        import tensorflow as tf
        """
        Calcula a matriz Hessiana exata (segundas derivadas parciais) usando diferenciação automática.

        ATENÇÃO: A função matemática passada no parâmetro 'funcao' DEVE ser escrita utilizando
        estritamente as operações e funções do TensorFlow (ex: tf.reduce_sum, tf.sin, tf.exp, etc.).
        Funções que utilizam NumPy (np.sum, np.sin) ou bibliotecas padrão (math.sin) causaram erros
        de rastreamento (AttributeError ou TypeError) porque o TensorFlow não consegue computar gradientes
        de operações executadas fora de seu grafo.

        Exemplo de uso correto:
        f = lambda x: tf.reduce_sum(tf.sin(x))
        GradienteMatrizHessiana(f, [1.0, 2.0, 3.0])

        Para o caso onde voce precisar aplicar uma funcão separadamente\n
        Onde as variaveis se cruzam, ou seja, calcular para cada valor de x, voce pode fazer por exmeplo:\n
        f = lambda x: x[0]**2 * x[1]**3
        hessiana = GradienteMatrizHessiana(f, x)
        onde x = [1.0, 2.0, 3.0]
        Parâmetros:
        ----------
        funcao: callable
            Função que recebe um Tensor do TensorFlow e retorna um escalar TensorFlow.
        x_vetor: array-like
            Vetor/ponto de avaliação das segundas derivadas.
        """
        # Convertemos o ponto x para um tensor que o TF possa monitorar
        x_tensor = tf.Variable(x_vetor, dtype=tf.float64)

        with tf.GradientTape() as tape_out:
            with tf.GradientTape() as tape_inner:
                y = funcao(x_tensor)

            # 1. Obtém o vetor gradiente (primeiras derivadas) dentro da fita externa
            gradiente = tape_inner.gradient(y, x_tensor)

        # 2. Deriva o gradiente em relação a x_tensor usando a fita externa
        hessiana = tape_out.jacobian(gradiente, x_tensor)

        return hessiana.numpy()
        
    def DerivadaDivergente3D(self, P, Q, R, x, y, z):
        """
        O divergente mede a taxa líquida de fluxo saindo de um determinado ponto infinitesimal de um campo vetorial.\n
        Ele indica se um ponto funciona como uma fonte (divergência positiva) ou um sumidouro (divergência negativa).\n
        A quem se aplica: A um campo vetorial, ou seja, produz um escalar.
        """
        NablaP = self.LaplacianoPrimeiraOrdem(P, x, y, z)
        NablaQ = self.LaplacianoPrimeiraOrdem(Q, y, x, z)
        NablaR = self.LaplacianoPrimeiraOrdem(R, z, x, y)

        return NablaP + NablaQ + NablaR

    def DerivadaRotacional3D(self, P, Q, R, x, y, z):
        """
        O rotacional mede a tendência de rotação (o "giro") de um campo vetorial em torno de um ponto. Ele aponta na direção do eixo de rotação de máxima circulação, seguindo a regra da mão direita.
        """
        # Isola e calcula cada derivada parcial unidimensionalmente de forma limpa e direta
        Ry = self.Derivada(lambda v: R(x, v, z), y)
        Py = self.Derivada(lambda v: P(x, v, z), y)
        
        Qz = self.Derivada(lambda v: Q(x, y, v), z)
        Pz = self.Derivada(lambda v: P(x, y, v), z)
        
        Rx = self.Derivada(lambda v: R(v, y, z), x)
        Qx = self.Derivada(lambda v: Q(v, y, z), x)

        return np.array([(Ry - Qz), (Pz - Rx), (Qx - Py)])

