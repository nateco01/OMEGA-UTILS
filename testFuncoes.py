from funcoes import FuncoesSolverNaoLinear, FuncoesSolver
import math
f = FuncoesSolverNaoLinear()

funcao = lambda x: math.cos(x) - x
res = f.MetodoFalsaPosicao(funcao, -1, 1, 50, 0.000001)
print(res)