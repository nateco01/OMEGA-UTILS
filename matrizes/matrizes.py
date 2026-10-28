import tensorflow as tf
import numpy as np
from functools import reduce

class MatrizesSolver():
    def __init__(self):
        pass
    
    def OperacoesMatricial(self, mode="soma", *n):
        """
        Com este metodo é possivel realizar operações de soma, subtração, multiplicação e divisão entre matrizes.\n
        Defina a quantidade de tensores *n e aplique a operção desejada\n
        Lembre-se, a quantidade de colunas da matriz A deve ser a mesma que a quantidade de linhas da matriz B especialmente no caso de multiplicação\n

        Parametros:
        soma: Soma as matrizes\n
        sub: Subtrai matrizes\n
        mult: Multiplica matrizes\n
        div: Divide matrizes. Obs: Não existe a divisão direta entre matrizes, é reciso realizar a operção inversa das matrizes e entao multiplica-las.
        """
        if not n:
            return None

        matrizes = tuple(tf.convert_to_tensor(matriz, dtype=tf.float64) for matriz in n)
        if mode == "soma":
            return tf.add_n(matrizes)

        if mode == "sub":
            return reduce(lambda x, y: tf.subtract(x, y), matrizes)
        
        if mode == "mult":
            return reduce(lambda x, y: tf.matmul(x, y), matrizes)

        if mode == "div":
            resultado = matrizes[0]
            for divisor in matrizes[1:]:
                resultado = tf.transpose(
                    tf.linalg.solve(tf.transpose(divisor), tf.transpose(resultado))
                )
            return resultado

        raise ValueError("mode deve ser 'soma', 'sub', 'mult' ou 'div'.")

    def SistemaLinear(self, A, b):
        """
        Resolve um sistema de equações lineares da forma Ax = b.
        
        Parâmetros:
            A: Matriz de coeficientes (quadrada).
            b: Vetor ou matriz de termos independentes.
            
        Retorna:
            O vetor solução x.
        """
        A_tensor = tf.convert_to_tensor(A, dtype=tf.float64)
        b_tensor = tf.convert_to_tensor(b, dtype=tf.float64)

        # Verifica se b é unidimensional para ajustar o formato exigido pelo solver
        is_1d = (b_tensor.shape.ndims == 1)
        if is_1d:
            b_tensor = tf.expand_dims(b_tensor, axis=-1)

        # Resolve o sistema linear Ax = b de forma eficiente
        x = tf.linalg.solve(A_tensor, b_tensor)

        # Retorna ao formato 1D original se a entrada era um vetor simples
        if is_1d:
            x = tf.squeeze(x, axis=-1)

        return x

    def Determinante(self, A):
        """Calcula o determinante de uma matriz quadrada A."""
        A_tensor = tf.convert_to_tensor(A, dtype=tf.float64)
        return tf.linalg.det(A_tensor)

    def Inversa(self, A):
        """Calcula a matriz inversa de uma matriz quadrada A."""
        A_tensor = tf.convert_to_tensor(A, dtype=tf.float64)
        return tf.linalg.inv(A_tensor)

    def AutovaloresAutovetores(self, A):
        """
        Calcula os autovalores e autovetores de uma matriz quadrada A.
        Retorna uma tupla (autovalores, autovetores).
        """
        A_tensor = tf.convert_to_tensor(A, dtype=tf.float64)
        autovalores, autovetores = tf.linalg.eig(A_tensor)
        return autovalores, autovetores

    def Norma(self, A, ord=None):
        """Calcula a norma de vetores ou matrizes."""
        return tf.linalg.norm(tf.convert_to_tensor(A, dtype=tf.float64), ord=ord)

    def SVD(self, A):
        """Calcula a Decomposição em Valores Singulares (SVD): A = U * S * V^T."""
        s, u, v = tf.linalg.svd(tf.convert_to_tensor(A, dtype=tf.float64))
        return s, u, v
    
    def GaussJordan(self, A, b):
        """
        Resolve Ax=b por eliminacao de Gauss-Jordan em float64 com pivoteamento
        parcial escalado. Retorna tensor TensorFlow; aceita um ou varios RHS.
        """
        coeficientes = np.asarray(A, dtype=np.float64)
        termos = np.asarray(b, dtype=np.float64)
        if coeficientes.ndim != 2 or coeficientes.shape[0] != coeficientes.shape[1]:
            raise ValueError("A deve ser uma matriz quadrada.")
        n = coeficientes.shape[0]
        vetor = termos.ndim == 1
        if vetor:
            termos = termos[:, None]
        if termos.ndim != 2 or termos.shape[0] != n:
            raise ValueError("b deve ter n linhas, como vetor ou matriz de termos independentes.")
        if not np.all(np.isfinite(coeficientes)) or not np.all(np.isfinite(termos)):
            raise ValueError("A e b devem conter apenas valores finitos.")

        aumentada = np.column_stack((coeficientes.copy(), termos.copy()))
        escalas_linhas = np.max(np.abs(coeficientes), axis=1)
        if np.any(escalas_linhas == 0.0):
            raise np.linalg.LinAlgError("A matriz possui uma linha nula e e singular.")
        fator_tolerancia = np.finfo(np.float64).eps * max(1, n)
        for coluna in range(n):
            razoes = np.abs(aumentada[coluna:, coluna]) / escalas_linhas[coluna:]
            linha_pivo = coluna + int(np.argmax(razoes))
            pivo = aumentada[linha_pivo, coluna]
            if abs(pivo) <= fator_tolerancia * escalas_linhas[linha_pivo]:
                raise np.linalg.LinAlgError("A matriz e singular ou numericamente singular.")
            if linha_pivo != coluna:
                aumentada[[coluna, linha_pivo]] = aumentada[[linha_pivo, coluna]]
                escalas_linhas[[coluna, linha_pivo]] = escalas_linhas[[linha_pivo, coluna]]
            aumentada[coluna] /= aumentada[coluna, coluna]
            for linha in range(n):
                if linha != coluna:
                    aumentada[linha] -= aumentada[linha, coluna] * aumentada[coluna]

        solucao = aumentada[:, n:]
        if vetor:
            solucao = solucao[:, 0]
        return tf.convert_to_tensor(solucao, dtype=tf.float64)

    def RegressaoLinear(self, X, y, mode="numpy"):
        """
        Resolve regressão linear: y = X * beta
        Retorna os coeficientes beta que minimizam ||y - X*beta||^2
        """
        if mode == "numpy":
            # Solução analítica direta (equações normais)
            XtX = X.T @ X
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)
            return beta
        
        if mode == "tensor":
            # Usando TensorFlow
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float64)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.float64)
            XtX = tf.matmul(X_tensor, X_tensor, transpose_a=True)
            Xty = tf.matmul(X_tensor, y_tensor, transpose_a=True)
            beta = tf.linalg.solve(XtX, Xty)
            return beta.numpy()