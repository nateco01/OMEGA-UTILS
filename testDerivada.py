from derivadas.derivadas import DerivadasSolver
from funcoes.funcoes import FuncoesSolver
import numpy as np

#f = FuncoesSolver()
d = DerivadasSolver()

# x = 1.5
# y = 5.5
# z = 54.6

# p = lambda x: f.HipoerbolicaSeno(x)
# q = lambda y: f.HipoerbolicaCosseno(y)
# r = lambda z: f.HipoerbolicaCotangente(z)

# res = d.DerivadaRotacional3D(p, q, r, 1.0, 5.5, 6.9)

# print(res)

# Simulação exata do teste do usuário usando funções reais 
# P(x,y,z) = sinh(x), Q(x,y,z) = cosh(y), R(x,y,z) = coth(z)

p_user = lambda x, y, z:np.sinh(x)

q_user = lambda x, y, z:np.cosh(y)

r_user = lambda x, y, z: np.cosh(z) / np.sinh(z) 

# Ponto solicitado pelo usuário: x=1.0, y=5.5, z=6.9
x_u, y_u, z_u = 1.0, 5.5, 6.9

rot_u = d.DerivadaRotacional3D(p_user, q_user, r_user, x_u, y_u, z_u)
print("Resultado correto esperado para o campo do usuário:")
print(rot_u)